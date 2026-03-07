"""Experience memory — routing lessons, reward model, skill self-generation, and MD sync."""

import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..logging import get_logger
from .memory import DB_PATH

_log = get_logger("experience")

# Lazy-loaded jieba for tokenization
_jieba = None
_jieba_load_failed = False


def _tokenize(text: str) -> set[str]:
    """Tokenize text using jieba (word-boundary segmentation). Falls back to char n-grams."""
    global _jieba, _jieba_load_failed
    if _jieba_load_failed:
        # Fallback: whitespace + individual chars for CJK
        return set(text.lower().split())
    if _jieba is None:
        try:
            import jieba
            jieba.setLogLevel(20)  # suppress loading messages
            _jieba = jieba
        except ImportError:
            _jieba_load_failed = True
            return set(text.lower().split())
    return set(w for w in _jieba.cut(text.lower()) if len(w.strip()) > 0)


# Maximum number of lessons allowed
MAX_LESSONS = 100

_DEFAULT_MD_PATH = Path.home() / ".liagent" / "experiences.md"

# ─── Skill generation prompt for Nemotron ──────────────────────────────────

SKILL_GEN_SYSTEM = """\
You are LiAgent's experience analyzer. Analyze why a user query failed and generate a routing lesson.

Output one JSON object:
{
  "pattern": "short pattern description",
  "keywords": ["keyword1", "keyword2"],
  "category": "realtime_price|realtime_search|file_op|code_exec|general",
  "should_use_tool": true,
  "suggested_tool": "tool name or empty string",
  "confidence": 0.7,
  "reasoning": "why this rule is needed"
}

Rules:
- keywords should generalize to similar queries (not too specific)
- confidence must be <= 0.9 for new rules
- if no tool is needed (for example casual chat), set should_use_tool=false
- max 15 keywords
- output JSON only
"""


@dataclass
class MatchResult:
    pattern: str
    keywords: list[str]
    category: str
    should_use_tool: bool
    suggested_tool: str
    confidence: float
    score: float
    lesson_id: int


@dataclass
class _Lesson:
    id: int
    pattern: str
    keywords: list[str]
    category: str
    should_use_tool: bool
    suggested_tool: str
    confidence: float
    success_count: int
    fail_count: int
    source: str


# ─── Seed experiences ──────────────────────────────────────────────────────

_SEED_LESSONS = [
    {
        "pattern": "price + asset",
        "keywords": ["price", "stock price", "quote", "gold", "silver", "oil", "bitcoin"],
        "category": "realtime_price",
        "should_use_tool": True,
        "suggested_tool": "web_search",
        "confidence": 0.8,
    },
    {
        "pattern": "ticker symbols",
        "keywords": ["AAPL", "TSLA", "MSFT", "GOOG", "GOOGL", "AMZN", "NVDA"],
        "category": "realtime_price",
        "should_use_tool": True,
        "suggested_tool": "web_search",
        "confidence": 0.8,
    },
    {
        "pattern": "company names + price",
        "keywords": ["google", "apple", "tesla", "microsoft", "amazon", "nvidia", "price"],
        "category": "realtime_price",
        "should_use_tool": True,
        "suggested_tool": "web_search",
        "confidence": 0.8,
    },
    {
        "pattern": "real-time info",
        "keywords": ["latest", "today", "current", "weather", "news", "score"],
        "category": "realtime_search",
        "should_use_tool": True,
        "suggested_tool": "web_search",
        "confidence": 0.7,
    },
    {
        "pattern": "explanatory question",
        "keywords": ["what is", "how", "why", "explain", "principle"],
        "category": "explanation",
        "should_use_tool": False,
        "suggested_tool": "",
        "confidence": 0.7,
    },
]


class ExperienceMemory:
    """Keyword-based experience matching with Laplace-smoothed reward scoring."""

    def __init__(
        self,
        db_path: Path = DB_PATH,
        md_path: Path = _DEFAULT_MD_PATH,
        journal=None,
    ):
        self.db_path = db_path
        self.md_path = md_path
        self.journal = journal
        self._last_skill_gen_ts: float = 0.0
        self._init_db()

    # ─── Schema ────────────────────────────────────────────────────────────

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS tool_routing_lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    category TEXT NOT NULL,
                    should_use_tool INTEGER NOT NULL DEFAULT 1,
                    suggested_tool TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    fail_count INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'seed',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            # Seed if empty
            count = conn.execute(
                "SELECT COUNT(*) FROM tool_routing_lessons"
            ).fetchone()[0]
            if count == 0:
                self._seed(conn)

    def _seed(self, conn: sqlite3.Connection):
        now = datetime.now(timezone.utc).isoformat()
        for s in _SEED_LESSONS:
            conn.execute(
                """INSERT INTO tool_routing_lessons
                   (pattern, keywords, category, should_use_tool, suggested_tool,
                    confidence, success_count, fail_count, source, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 0, 0, 'seed', ?, ?)""",
                (
                    s["pattern"],
                    json.dumps(s["keywords"], ensure_ascii=False),
                    s["category"],
                    int(s["should_use_tool"]),
                    s["suggested_tool"],
                    s["confidence"],
                    now,
                    now,
                ),
            )

    # ─── CRUD ──────────────────────────────────────────────────────────────

    def _all_lessons(self) -> list[_Lesson]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, pattern, keywords, category, should_use_tool, "
                "suggested_tool, confidence, success_count, fail_count, source "
                "FROM tool_routing_lessons ORDER BY id"
            ).fetchall()
        lessons = []
        for r in rows:
            try:
                kw = json.loads(r[2])
            except (json.JSONDecodeError, TypeError):
                kw = []
            lessons.append(
                _Lesson(
                    id=r[0],
                    pattern=r[1],
                    keywords=[str(k).lower() for k in kw],
                    category=r[3],
                    should_use_tool=bool(r[4]),
                    suggested_tool=r[5],
                    confidence=float(r[6]),
                    success_count=int(r[7]),
                    fail_count=int(r[8]),
                    source=r[9],
                )
            )
        return lessons

    def _find_overlapping_lesson(self, keywords: list[str], threshold: float = 0.5) -> int | None:
        """Find existing lesson with >threshold keyword overlap. Returns lesson id or None."""
        new_kw = set(k.lower() for k in keywords)
        if not new_kw:
            return None
        for lesson in self._all_lessons():
            existing_kw = set(lesson.keywords)
            if not existing_kw:
                continue
            overlap = len(new_kw & existing_kw) / max(len(new_kw), len(existing_kw))
            if overlap > threshold:
                return lesson.id
        return None

    def _enforce_capacity(self):
        """Remove lowest-confidence non-seed lessons if over MAX_LESSONS."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM tool_routing_lessons").fetchone()[0]
            if count <= MAX_LESSONS:
                return
            # Remove lowest-confidence non-seed lessons
            excess = count - MAX_LESSONS
            conn.execute(
                """DELETE FROM tool_routing_lessons WHERE id IN (
                    SELECT id FROM tool_routing_lessons
                    WHERE source != 'seed'
                    ORDER BY confidence ASC, (success_count + fail_count) ASC
                    LIMIT ?
                )""",
                (excess,),
            )

    def add_lesson(
        self,
        pattern: str,
        keywords: list[str],
        category: str,
        should_use_tool: bool = True,
        suggested_tool: str = "",
        confidence: float = 0.7,
        source: str = "nemotron_gen",
    ):
        """Add a new routing lesson to the database.

        If >50% keyword overlap with an existing lesson, merges instead of adding.
        Enforces MAX_LESSONS capacity limit.
        """
        keywords = keywords[:15]
        confidence = max(0.1, min(0.9, float(confidence)))
        now = datetime.now(timezone.utc).isoformat()

        # Check for keyword overlap — merge if >50%
        overlap_id = self._find_overlapping_lesson(keywords)
        if overlap_id is not None:
            with sqlite3.connect(self.db_path) as conn:
                # Merge: update confidence (take max) and add new keywords
                existing = conn.execute(
                    "SELECT keywords, confidence FROM tool_routing_lessons WHERE id = ?",
                    (overlap_id,),
                ).fetchone()
                if existing:
                    try:
                        old_kw = json.loads(existing[0])
                    except (json.JSONDecodeError, TypeError):
                        old_kw = []
                    merged_kw = list(set(old_kw + keywords))[:15]
                    merged_conf = max(float(existing[1]), confidence)
                    conn.execute(
                        "UPDATE tool_routing_lessons SET keywords = ?, confidence = ?, updated_at = ? WHERE id = ?",
                        (json.dumps(merged_kw, ensure_ascii=False), merged_conf, now, overlap_id),
                    )
                    return
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO tool_routing_lessons
                   (pattern, keywords, category, should_use_tool, suggested_tool,
                    confidence, success_count, fail_count, source, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)""",
                (
                    pattern,
                    json.dumps(keywords, ensure_ascii=False),
                    category,
                    int(should_use_tool),
                    suggested_tool,
                    confidence,
                    source,
                    now,
                    now,
                ),
            )
        self._enforce_capacity()

    # ─── Matching ──────────────────────────────────────────────────────────

    def _keyword_doc_freq(self, lessons: list[_Lesson]) -> dict[str, int]:
        """Count how many lessons each keyword appears in (for IDF weighting)."""
        freq: dict[str, int] = {}
        for lesson in lessons:
            for kw in lesson.keywords:
                freq[kw] = freq.get(kw, 0) + 1
        return freq

    def match(self, query: str) -> MatchResult | None:
        """Find the best-matching lesson for a query using tokenized IDF-weighted scoring.

        Uses jieba segmentation for word-boundary matching.
        IDF weighting ensures rare keywords score higher than common ones.
        """
        q = (query or "").lower()
        if not q:
            return None

        all_lessons = self._all_lessons()
        if not all_lessons:
            return None

        q_tokens = _tokenize(q)
        kw_doc_freq = self._keyword_doc_freq(all_lessons)
        num_lessons = len(all_lessons)

        candidates: list[tuple[float, _Lesson]] = []
        for lesson in all_lessons:
            kw_set = set(lesson.keywords)
            hits = q_tokens & kw_set  # set intersection (word boundary)
            if not hits:
                continue
            # IDF-weighted score: rare keywords contribute more
            idf_score = sum(
                1.0 / max(1, kw_doc_freq.get(kw, 1))
                for kw in hits
            )
            reward = self.get_reward_score(lesson.category, lesson.suggested_tool)
            score = idf_score * lesson.confidence * reward
            candidates.append((score, lesson))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best = candidates[0]
        if best_score < 0.5:
            return None

        return MatchResult(
            pattern=best.pattern,
            keywords=best.keywords,
            category=best.category,
            should_use_tool=best.should_use_tool,
            suggested_tool=best.suggested_tool,
            confidence=best.confidence,
            score=best_score,
            lesson_id=best.id,
        )

    # ─── Reward model ─────────────────────────────────────────────────────

    def get_reward_score(self, category: str, tool_name: str) -> float:
        """Laplace-smoothed success rate, scaled to [0, 2.0]."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT COALESCE(SUM(success_count), 0), COALESCE(SUM(fail_count), 0)
                   FROM tool_routing_lessons
                   WHERE category = ? AND suggested_tool = ?""",
                (category, tool_name),
            ).fetchone()
        success = int(row[0]) if row else 0
        fail = int(row[1]) if row else 0
        return (success + 1) / (success + fail + 2) * 2.0

    # ─── Outcome recording ─────────────────────────────────────────────────

    def record_outcome(
        self,
        query: str,
        tool_used: str | None,
        success: bool,
        source: str = "self_eval",
    ):
        """Record success/failure for matched lessons to update reward scores."""
        match = self.match(query)
        if match is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        col = "success_count" if success else "fail_count"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE tool_routing_lessons SET {col} = {col} + 1, updated_at = ? WHERE id = ?",
                (now, match.lesson_id),
            )
            if self.journal:
                row = conn.execute(
                    "SELECT success_count, fail_count FROM tool_routing_lessons WHERE id = ?",
                    (match.lesson_id,),
                ).fetchone()
                if row:
                    s, f = row
                    reward = (s + 1) / (s + f + 2) * 2.0
                    self.journal.outcome_recorded(
                        query=query, tool=tool_used, success=success,
                        source=source, reward=reward,
                    )

    # ─── Pruning ───────────────────────────────────────────────────────────

    def prune_stale(self, min_confidence: float = 0.2):
        """Remove lessons with very low confidence (excluding seeds with no data)."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """DELETE FROM tool_routing_lessons
                   WHERE confidence < ?
                     AND source != 'seed'
                     AND (success_count + fail_count) > 3""",
                (min_confidence,),
            )
            if self.journal:
                self.journal.experience_pruned(deleted_count=cur.rowcount)

    # ─── Markdown sync ─────────────────────────────────────────────────────

    def sync_to_markdown(self):
        """Write current SQLite lessons to experiences.md for human review."""
        lessons = self._all_lessons()
        self.md_path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["# LiAgent Experience Rules\n"]
        for ls in lessons:
            lines.append(f"## {ls.pattern}")
            lines.append(f"- **pattern**: {ls.pattern}")
            lines.append(f"- **keywords**: {', '.join(ls.keywords)}")
            lines.append(f"- **category**: {ls.category}")
            lines.append(f"- **tool**: {ls.suggested_tool or '(none)'}")
            lines.append(f"- **should_use_tool**: {ls.should_use_tool}")
            lines.append(f"- **confidence**: {ls.confidence}")
            lines.append(f"- **source**: {ls.source}")
            lines.append(
                f"- **stats**: {ls.success_count} success / {ls.fail_count} failure"
            )
            lines.append("")

        self.md_path.write_text("\n".join(lines), encoding="utf-8")

    def sync_from_markdown(self):
        """Read experiences.md and merge user edits back into SQLite.

        User-edited entries (not matching any existing pattern) get source='user_edit'
        with high confidence. Existing entries get their confidence/keywords updated.
        """
        if not self.md_path.exists():
            return

        text = self.md_path.read_text(encoding="utf-8")
        blocks = re.split(r"^## ", text, flags=re.MULTILINE)
        existing = {ls.pattern: ls for ls in self._all_lessons()}

        for block in blocks[1:]:  # skip header
            parsed = self._parse_md_block(block)
            if not parsed:
                continue
            pattern = parsed["pattern"]
            if pattern in existing:
                # Update confidence/keywords if changed
                ls = existing[pattern]
                new_conf = parsed.get("confidence", ls.confidence)
                new_kw = parsed.get("keywords", ls.keywords)
                if new_conf != ls.confidence or new_kw != ls.keywords:
                    now = datetime.now(timezone.utc).isoformat()
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute(
                            """UPDATE tool_routing_lessons
                               SET confidence = ?, keywords = ?, updated_at = ?
                               WHERE id = ?""",
                            (
                                new_conf,
                                json.dumps(new_kw, ensure_ascii=False),
                                now,
                                ls.id,
                            ),
                        )
            else:
                # New user-added entry
                self.add_lesson(
                    pattern=pattern,
                    keywords=parsed.get("keywords", []),
                    category=parsed.get("category", "general"),
                    should_use_tool=parsed.get("should_use_tool", True),
                    suggested_tool=parsed.get("tool", ""),
                    confidence=min(0.95, parsed.get("confidence", 0.8)),
                    source="user_edit",
                )

    @staticmethod
    def _parse_md_block(block: str) -> dict | None:
        """Parse a single ## block from experiences.md."""
        lines = block.strip().split("\n")
        if not lines:
            return None
        result: dict = {"pattern": lines[0].strip()}
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("- **pattern**:"):
                result["pattern"] = line.split(":", 1)[1].strip()
            elif line.startswith("- **keywords**:"):
                raw = line.split(":", 1)[1].strip()
                result["keywords"] = [k.strip().lower() for k in raw.split(",") if k.strip()]
            elif line.startswith("- **category**:"):
                result["category"] = line.split(":", 1)[1].strip()
            elif line.startswith("- **tool**:"):
                val = line.split(":", 1)[1].strip()
                result["tool"] = "" if val == "(none)" else val
            elif line.startswith("- **should_use_tool**:"):
                val = line.split(":", 1)[1].strip().lower()
                result["should_use_tool"] = val in ("true", "1", "yes")
            elif line.startswith("- **confidence**:"):
                try:
                    result["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        return result if result.get("pattern") else None

    # ─── Skill self-generation (Nemotron) ──────────────────────────────────

    async def generate_skill(
        self,
        engine,
        query: str,
        failed_answer: str,
        tool_used: str | None,
        available_tools_summary: str,
    ) -> bool:
        """Use Nemotron 30B to analyze a failure and generate a new routing lesson.

        Returns True if a new lesson was created, False otherwise.
        Rate-limited to once per 60 seconds.
        """
        now = time.time()
        if now - self._last_skill_gen_ts < 60:
            return False

        # Only generate if no existing match
        if self.match(query) is not None:
            return False

        self._last_skill_gen_ts = now

        prompt = [
            {"role": "system", "content": SKILL_GEN_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"User query: {query}\n"
                    f"Agent answer: {failed_answer[:500]}\n"
                    f"Tool used: {tool_used or 'none'}\n"
                    f"User feedback: negative\n"
                    f"Available tools: {available_tools_summary}\n"
                    "Analyze this failure and generate one new routing lesson."
                ),
            },
        ]

        try:
            result = await engine.generate_reasoning(
                prompt, max_tokens=512, temperature=0.3,
                enable_thinking=False,
            )
            lesson = self._parse_skill_gen_result(result)
            if lesson:
                self.add_lesson(**lesson, source="nemotron_gen")
                self.sync_to_markdown()
                if self.journal:
                    self.journal.skill_generated(
                        query=query, failed_answer=failed_answer,
                        pattern=lesson.get("pattern", ""),
                        tool=lesson.get("suggested_tool", ""),
                        confidence=lesson.get("confidence", 0),
                        source="nemotron_gen",
                    )
                return True
        except Exception as exc:
            _log.warning("skill_gen_error", error=str(exc))
        return False

    @staticmethod
    def _parse_skill_gen_result(raw: str) -> dict | None:
        """Parse Nemotron JSON output into lesson kwargs."""
        # Try direct parse
        for text in [raw.strip(), None]:
            if text is None:
                # Regex fallback
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if not m:
                    return None
                text = m.group()
            try:
                obj = json.loads(text)
                if not isinstance(obj, dict):
                    continue
                pattern = str(obj.get("pattern", "")).strip()
                keywords = obj.get("keywords", [])
                category = str(obj.get("category", "general")).strip()
                if not pattern or not keywords:
                    continue
                if not isinstance(keywords, list):
                    continue
                keywords = [str(k).strip().lower() for k in keywords[:15] if str(k).strip()]
                if len(keywords) < 2:
                    continue
                confidence = max(0.6, min(0.9, float(obj.get("confidence", 0.7))))
                return {
                    "pattern": pattern,
                    "keywords": keywords,
                    "category": category,
                    "should_use_tool": bool(obj.get("should_use_tool", True)),
                    "suggested_tool": str(obj.get("suggested_tool", "")).strip(),
                    "confidence": confidence,
                }
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return None
