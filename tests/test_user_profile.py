"""Tests for UserProfileStore (user_profile_slots table)."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from liagent.agent.memory import LongTermMemory, UserProfileStore
from liagent.agent.prompt_builder import PromptBuilder


class ProfileStoreBasicTests(unittest.TestCase):
    """CRUD operations on user_profile_slots."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.db_path = Path(self.td.name) / "test.db"
        self.ltm = LongTermMemory(
            db_path=self.db_path, data_dir=Path(self.td.name) / "data"
        )
        self.store = UserProfileStore(self.db_path)

    def tearDown(self):
        self.td.cleanup()

    def test_table_exists(self):
        with sqlite3.connect(self.db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        self.assertIn("user_profile_slots", tables)

    def test_get_empty(self):
        slot = self.store.get("language")
        self.assertIsNone(slot)

    def test_get_all_empty(self):
        self.assertEqual(self.store.get_all(), [])

    def test_upsert_and_get(self):
        self.store.upsert(
            "language", "zh", confidence=0.8, source="implicit"
        )
        slot = self.store.get("language")
        self.assertIsNotNone(slot)
        self.assertEqual(slot["dimension"], "language")
        self.assertEqual(slot["value"], "zh")
        self.assertAlmostEqual(slot["confidence"], 0.8)
        self.assertEqual(slot["source"], "implicit")
        self.assertEqual(slot["locked"], 0)

    def test_upsert_overwrites(self):
        self.store.upsert("language", "zh", confidence=0.5, source="implicit")
        self.store.upsert("language", "en", confidence=0.9, source="user_stated")
        slot = self.store.get("language")
        self.assertEqual(slot["value"], "en")
        self.assertAlmostEqual(slot["confidence"], 0.9)

    def test_delete(self):
        self.store.upsert("language", "zh", confidence=0.5, source="implicit")
        self.store.delete("language")
        self.assertIsNone(self.store.get("language"))

    def test_delete_nonexistent_is_noop(self):
        self.store.delete("language")  # should not raise

    def test_delete_all(self):
        self.store.upsert("language", "zh", confidence=0.5, source="implicit")
        self.store.upsert("tone", "casual", confidence=0.6, source="implicit")
        self.store.delete_all()
        self.assertEqual(self.store.get_all(), [])

    def test_get_all_returns_all(self):
        self.store.upsert("language", "zh", confidence=0.5, source="implicit")
        self.store.upsert("tone", "casual", confidence=0.6, source="implicit")
        slots = self.store.get_all()
        dims = {s["dimension"] for s in slots}
        self.assertEqual(dims, {"language", "tone"})


class NormalizeDomainsTests(unittest.TestCase):
    def test_dedup_and_sort(self):
        self.assertEqual(
            UserProfileStore.normalize_domains("AI, semiconductors ,ai"),
            "ai,semiconductors",
        )

    def test_empty(self):
        self.assertEqual(UserProfileStore.normalize_domains(""), "")

    def test_single(self):
        self.assertEqual(UserProfileStore.normalize_domains("crypto"), "crypto")


class MergeImplicitTests(unittest.TestCase):
    """Hysteresis merge state machine — design doc section 4."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.db_path = Path(self.td.name) / "test.db"
        self.ltm = LongTermMemory(
            db_path=self.db_path, data_dir=Path(self.td.name) / "data"
        )
        self.store = UserProfileStore(self.db_path)

    def tearDown(self):
        self.td.cleanup()

    def test_new_dimension(self):
        """Case 1: slot doesn't exist → create with 0.3 + nudge."""
        self.store.merge_implicit("language", "zh", "moderate")
        slot = self.store.get("language")
        self.assertEqual(slot["value"], "zh")
        self.assertAlmostEqual(slot["confidence"], 0.38)  # 0.3 + 0.08
        self.assertEqual(slot["evidence_count"], 1)
        self.assertEqual(slot["source"], "implicit")

    def test_same_direction_reinforcement(self):
        """Case 2: same value → confidence increases."""
        for _ in range(3):
            self.store.merge_implicit("language", "zh", "moderate")
        slot = self.store.get("language")
        self.assertEqual(slot["value"], "zh")
        # 0.38 + 0.08 + 0.08 = 0.54
        self.assertAlmostEqual(slot["confidence"], 0.54)
        self.assertEqual(slot["evidence_count"], 3)

    def test_single_opposing_signal(self):
        """Case 3: opposing value → confidence decreases, candidate set."""
        self.store.merge_implicit("language", "zh", "moderate")
        self.store.merge_implicit("language", "en", "moderate")
        slot = self.store.get("language")
        self.assertEqual(slot["value"], "zh")  # NOT flipped
        self.assertLess(slot["confidence"], 0.38)  # decreased
        self.assertEqual(slot["candidate_value"], "en")
        self.assertEqual(slot["candidate_evidence_count"], 1)

    def test_opposing_below_flip_threshold(self):
        """2 opposing signals → still no flip (need 3 + confidence < 0.3)."""
        self.store.merge_implicit("language", "zh", "moderate")
        self.store.merge_implicit("language", "en", "moderate")
        self.store.merge_implicit("language", "en", "moderate")
        slot = self.store.get("language")
        self.assertEqual(slot["value"], "zh")  # still zh
        self.assertEqual(slot["candidate_evidence_count"], 2)

    def test_flip_after_3_consistent_opposing(self):
        """Case 3 → flip: 3 consistent opposing + confidence < 0.3."""
        # Create with weak initial signal so confidence starts low
        self.store.merge_implicit("language", "zh", "weak")  # 0.34
        # 3 opposing signals
        self.store.merge_implicit("language", "en", "moderate")
        self.store.merge_implicit("language", "en", "moderate")
        self.store.merge_implicit("language", "en", "moderate")
        slot = self.store.get("language")
        self.assertEqual(slot["value"], "en")  # flipped!
        self.assertEqual(slot["candidate_value"], "")  # reset
        self.assertEqual(slot["candidate_evidence_count"], 0)

    def test_mixed_opposing_no_flip(self):
        """Mixed opposing directions → candidate resets, no flip."""
        self.store.merge_implicit("language", "zh", "weak")  # 0.34
        self.store.merge_implicit("language", "en", "moderate")   # candidate=en, count=1
        self.store.merge_implicit("language", "mixed", "moderate")  # different! candidate=mixed, count=1
        self.store.merge_implicit("language", "en", "moderate")   # different from mixed! candidate=en, count=1
        slot = self.store.get("language")
        self.assertEqual(slot["value"], "zh")  # no flip — candidate never reached 3
        self.assertEqual(slot["candidate_value"], "en")
        self.assertEqual(slot["candidate_evidence_count"], 1)

    def test_locked_slot_ignores_implicit(self):
        """Case 4: locked slot → no-op."""
        self.store.upsert(
            "language", "zh", confidence=0.95, source="user_stated", locked=1
        )
        self.store.merge_implicit("language", "en", "strong")
        slot = self.store.get("language")
        self.assertEqual(slot["value"], "zh")
        self.assertAlmostEqual(slot["confidence"], 0.95)

    def test_confidence_never_negative(self):
        """Confidence must never go below 0.0."""
        self.store.upsert("language", "zh", confidence=0.05, source="implicit")
        self.store.merge_implicit("language", "en", "strong")
        slot = self.store.get("language")
        self.assertGreaterEqual(slot["confidence"], 0.0)

    def test_confidence_never_above_one(self):
        """Confidence must never exceed 1.0."""
        self.store.upsert("language", "zh", confidence=0.98, source="implicit")
        self.store.merge_implicit("language", "zh", "strong")
        slot = self.store.get("language")
        self.assertLessEqual(slot["confidence"], 1.0)

    def test_domains_normalized_on_merge(self):
        """domains dimension should be normalized on merge."""
        self.store.merge_implicit("domains", "AI, crypto ,ai", "moderate")
        slot = self.store.get("domains")
        self.assertEqual(slot["value"], "ai,crypto")


class SetExplicitTests(unittest.TestCase):
    """Explicit preference commands — design doc section 3b."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.db_path = Path(self.td.name) / "test.db"
        self.ltm = LongTermMemory(
            db_path=self.db_path, data_dir=Path(self.td.name) / "data"
        )
        self.store = UserProfileStore(self.db_path)

    def tearDown(self):
        self.td.cleanup()

    def test_set_explicit_creates_locked_slot(self):
        self.store.set_explicit("language", "en")
        slot = self.store.get("language")
        self.assertEqual(slot["value"], "en")
        self.assertAlmostEqual(slot["confidence"], 0.95)
        self.assertEqual(slot["source"], "user_stated")
        self.assertEqual(slot["locked"], 1)

    def test_set_explicit_overwrites_locked(self):
        """New user_stated overwrites existing locked slot."""
        self.store.set_explicit("language", "zh")
        self.store.set_explicit("language", "en")
        slot = self.store.get("language")
        self.assertEqual(slot["value"], "en")
        self.assertEqual(slot["locked"], 1)

    def test_set_explicit_overwrites_implicit(self):
        self.store.merge_implicit("language", "zh", "strong")
        self.store.set_explicit("language", "en")
        slot = self.store.get("language")
        self.assertEqual(slot["value"], "en")
        self.assertEqual(slot["locked"], 1)
        self.assertAlmostEqual(slot["confidence"], 0.95)

    def test_set_explicit_normalizes_domains(self):
        self.store.set_explicit("domains", "AI, crypto ,ai")
        slot = self.store.get("domains")
        self.assertEqual(slot["value"], "ai,crypto")

    def test_forget_specific_dimension(self):
        self.store.set_explicit("language", "en")
        self.store.forget("language")
        self.assertIsNone(self.store.get("language"))

    def test_forget_all(self):
        self.store.set_explicit("language", "en")
        self.store.set_explicit("tone", "casual")
        self.store.merge_implicit("domains", "ai", "moderate")
        self.store.forget_all()
        self.assertEqual(self.store.get_all(), [])

    def test_forget_nonexistent_is_noop(self):
        self.store.forget("nonexistent")  # should not raise


class CompilePortraitTests(unittest.TestCase):
    """Portrait compilation — design doc section 5."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.db_path = Path(self.td.name) / "test.db"
        self.ltm = LongTermMemory(
            db_path=self.db_path, data_dir=Path(self.td.name) / "data"
        )
        self.store = UserProfileStore(self.db_path)

    def tearDown(self):
        self.td.cleanup()

    def test_empty_profile_returns_empty_string(self):
        self.assertEqual(self.store.compile_portrait(), "")

    def test_hard_preference_uses_requires(self):
        self.store.set_explicit("language", "zh")
        portrait = self.store.compile_portrait()
        self.assertIn("[User Profile]", portrait)
        self.assertIn("requires", portrait.lower())

    def test_soft_preference_uses_tends(self):
        self.store.merge_implicit("tone", "casual", "moderate")
        portrait = self.store.compile_portrait()
        self.assertIn("tends to", portrait.lower())

    def test_mixed_hard_soft(self):
        self.store.set_explicit("language", "zh")  # hard
        self.store.merge_implicit("tone", "casual", "moderate")  # soft
        portrait = self.store.compile_portrait()
        self.assertIn("requires", portrait.lower())
        self.assertIn("tends to", portrait.lower())

    def test_top5_cap(self):
        """Only top 5 by confidence should be included."""
        for i, dim in enumerate(
            ["language", "tone", "response_style", "domains",
             "expertise_level", "data_preference"]
        ):
            self.store.upsert(
                dim, f"val{i}",
                confidence=0.9 - i * 0.05,
                source="implicit",
            )
        portrait = self.store.compile_portrait()
        # 6 dimensions but only 5 should appear
        self.assertNotIn("data_preference", portrait)

    def test_low_confidence_excluded(self):
        """Slots with confidence <= 0.3 are excluded."""
        self.store.upsert("language", "zh", confidence=0.25, source="implicit")
        self.assertEqual(self.store.compile_portrait(), "")


class PromptInjectionTests(unittest.TestCase):
    """Portrait injection via _long_term_context() — design doc section 5."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.db_path = Path(self.td.name) / "test.db"
        self.ltm = LongTermMemory(
            db_path=self.db_path, data_dir=Path(self.td.name) / "data"
        )
        self.store = UserProfileStore(self.db_path)
        self.pb = PromptBuilder(self.ltm)

    def tearDown(self):
        self.td.cleanup()

    def test_empty_profile_no_block(self):
        ctx = self.pb._long_term_context()
        self.assertNotIn("[User Profile]", ctx)

    def test_profile_injected_in_context(self):
        self.store.set_explicit("language", "zh")
        ctx = self.pb._long_term_context()
        self.assertIn("[User Profile]", ctx)
        self.assertIn("requires", ctx.lower())

    def test_profile_injected_in_vlm_prompt(self):
        self.store.set_explicit("language", "zh")
        prompt = self.pb.build_system_prompt(query="test")
        self.assertIn("[User Profile]", prompt)

    def test_profile_injected_in_coder_prompt(self):
        self.store.set_explicit("language", "zh")
        prompt = self.pb.build_system_prompt_for_coder(query="test")
        self.assertIn("[User Profile]", prompt)

    def test_profile_injected_in_api_prompt(self):
        self.store.set_explicit("language", "zh")
        prompt = self.pb.build_system_prompt_for_api(query="test")
        self.assertIn("[User Profile]", prompt)

    def test_profile_appears_before_facts(self):
        """Portrait should appear before ## User Facts."""
        self.store.set_explicit("language", "zh")
        self.ltm.save_facts([
            {"fact": "user lives in Shanghai", "category": "location",
             "confidence": 0.9, "source": "llm_extract"}
        ])
        ctx = self.pb._long_term_context()
        profile_pos = ctx.find("[User Profile]")
        facts_pos = ctx.find("## User Facts")
        self.assertGreater(facts_pos, profile_pos)

    def test_sensitive_personal_facts_are_not_injected(self):
        self.ltm.save_facts([
            {"fact": "The user's preferred name is Alex.", "category": "personal",
             "confidence": 0.95, "source": "llm_extract"},
            {"fact": "user lives in Seattle", "category": "location",
             "confidence": 0.90, "source": "llm_extract"},
        ])
        ctx = self.pb._long_term_context()
        self.assertNotIn("preferred name", ctx.lower())
        self.assertNotIn("alex", ctx.lower())
        self.assertIn("user lives in Seattle", ctx)


class ExplicitDetectionRegexTests(unittest.TestCase):
    """First gate — regex matching. Design doc section 3b."""

    def test_english_remember(self):
        from liagent.agent.brain import _detect_profile_command
        self.assertTrue(_detect_profile_command("remember that I prefer concise answers"))

    def test_english_remember_my(self):
        from liagent.agent.brain import _detect_profile_command
        self.assertTrue(_detect_profile_command("remember my preference for English replies"))

    def test_english_forget_preference(self):
        from liagent.agent.brain import _detect_profile_command
        self.assertTrue(_detect_profile_command("forget that I prefer dark mode"))

    def test_bare_always_no_trigger(self):
        from liagent.agent.brain import _detect_profile_command
        self.assertFalse(_detect_profile_command("always use English"))

    def test_bare_cancel_no_trigger(self):
        from liagent.agent.brain import _detect_profile_command
        self.assertFalse(_detect_profile_command("cancel my order"))

    def test_bare_never_no_trigger(self):
        from liagent.agent.brain import _detect_profile_command
        self.assertFalse(_detect_profile_command("never mind"))

    def test_normal_task_no_trigger(self):
        from liagent.agent.brain import _detect_profile_command
        self.assertFalse(_detect_profile_command("summarize this article"))

    def test_mixed_command_plus_task(self):
        from liagent.agent.brain import _detect_profile_command
        self.assertTrue(
            _detect_profile_command("remember that I prefer concise answers, then summarize this article")
        )


class ConfirmProfileRegexTests(unittest.TestCase):
    """Confirmation regex for forget_all guard."""

    def test_confirm_keyword(self):
        from liagent.agent.brain import _CONFIRM_PROFILE_RE
        self.assertTrue(_CONFIRM_PROFILE_RE.match("confirm"))

    def test_yes_keyword(self):
        from liagent.agent.brain import _CONFIRM_PROFILE_RE
        self.assertTrue(_CONFIRM_PROFILE_RE.match("yes"))

    def test_ok_keyword(self):
        from liagent.agent.brain import _CONFIRM_PROFILE_RE
        self.assertTrue(_CONFIRM_PROFILE_RE.match("ok"))

    def test_okay_keyword(self):
        from liagent.agent.brain import _CONFIRM_PROFILE_RE
        self.assertTrue(_CONFIRM_PROFILE_RE.match("okay"))

    def test_normal_input_no_match(self):
        from liagent.agent.brain import _CONFIRM_PROFILE_RE
        self.assertIsNone(_CONFIRM_PROFILE_RE.match("check the stock price for me"))


class StripMarkdownFencesTests(unittest.TestCase):
    """Fence stripping for LLM JSON output."""

    def test_strips_json_fence(self):
        from liagent.agent.brain import _strip_markdown_fences
        raw = '```json\n{"action": "set"}\n```'
        self.assertEqual(_strip_markdown_fences(raw), '{"action": "set"}')

    def test_no_fence_passthrough(self):
        from liagent.agent.brain import _strip_markdown_fences
        raw = '{"action": "none"}'
        self.assertEqual(_strip_markdown_fences(raw), '{"action": "none"}')

    def test_bare_fence(self):
        from liagent.agent.brain import _strip_markdown_fences
        raw = '```\n{"action": "set"}\n```'
        self.assertEqual(_strip_markdown_fences(raw), '{"action": "set"}')


class BrainProfileIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Integration: brain.run() explicit path yields profile_update events."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self._db_path = Path(self._td.name) / "test.db"

    def tearDown(self):
        self._td.cleanup()

    def _make_brain(self):
        """Build AgentBrain with mocks — follows test_core.py pattern."""
        from unittest.mock import patch, MagicMock
        from liagent.agent.brain import AgentBrain
        from liagent.config import AppConfig

        db_path = self._db_path

        class _FakeEngine:
            def __init__(self):
                self.config = AppConfig()
                self.config.llm.max_tokens = 256
                self.config.llm.temperature = 0.2
                self.config.tool_profile = "research"
                self.tool_parser = MagicMock()
                self.tool_parser.parse.return_value = None

            async def generate_llm_routed(self, *a, **kw):
                yield "OK, noted."

            async def generate_llm(self, *a, **kw):
                yield "OK"

            async def generate_text(self, *a, **kw):
                yield "OK"

            async def stream_text(self, *a, **kw):
                yield "OK"

            async def generate_extraction(self, msgs, **kw):
                return '{"action": "set", "dimension": "language", "value": "en"}'

            async def generate_reasoning(self, *a, **kw):
                return "OK"

        class _FakeLTM:
            def __init__(self, **kw): pass
            def get_recent_summaries(self, *a, **kw): return []
            def get_all_facts(self, *a, **kw): return []
            def get_relevant_facts(self, *a, **kw): return []
            def get_relevant_evidence(self, *a, **kw): return []
            def decay_confidence(self, *a, **kw): return None
            def prune_memory(self, *a, **kw): return None
            def prune_old_records(self, *a, **kw): return None
            def save_feedback(self, *a, **kw): return None
            def apply_source_confidence(self, facts): return facts
            def detect_conflicts(self, *a, **kw): return None

        _FakeLTM.db_path = db_path

        class _FakeExperience:
            def __init__(self, *a, **kw): pass
            def match(self, q): return None
            def record_outcome(self, *a, **kw): pass
            def sync_from_markdown(self): pass
            def sync_to_markdown(self): pass

        class _FakeMetrics:
            def __init__(self): pass
            def record(self, *a, **kw): pass
            def log_turn(self, *a, **kw): pass

        with patch("liagent.agent.brain.LongTermMemory", _FakeLTM), \
             patch("liagent.agent.brain.InteractionMetrics", _FakeMetrics), \
             patch("liagent.agent.brain.ExperienceMemory", _FakeExperience):
            brain = AgentBrain(_FakeEngine())
        return brain

    async def test_set_yields_profile_update(self):
        brain = self._make_brain()
        events = []
        async for ev in brain.run("remember that I prefer English", low_latency=True):
            events.append(ev)
        event_types = [e[0] for e in events]
        self.assertIn("profile_update", event_types)
        profile_ev = [e for e in events if e[0] == "profile_update"][0]
        self.assertIn("language", profile_ev[1])

    async def test_forget_all_yields_confirmation(self):
        """forget_all should yield confirmation prompt, NOT execute immediately."""
        brain = self._make_brain()
        async def fake_forget_all(msgs, **kw):
            return '{"action": "forget_all"}'
        brain.engine.generate_extraction = fake_forget_all
        brain.profile_store.set_explicit("language", "zh")
        events = []
        async for ev in brain.run("forget all my preferences", low_latency=True):
            events.append(ev)
        profile_evs = [e for e in events if e[0] == "profile_update"]
        self.assertTrue(len(profile_evs) > 0)
        self.assertIn("confirm", profile_evs[0][1].lower())
        # Slot should NOT be wiped yet
        slot = brain.profile_store.get("language")
        self.assertIsNotNone(slot)
        self.assertTrue(brain._pending_profile_forget_all)

    async def test_forget_all_executes_after_confirm(self):
        """After the confirmation prompt, user reply 'confirm' should execute forget_all."""
        brain = self._make_brain()
        brain.profile_store.set_explicit("language", "zh")
        brain._pending_profile_forget_all = True
        events = []
        async for ev in brain.run("confirm", low_latency=True):
            events.append(ev)
        profile_evs = [e for e in events if e[0] == "profile_update"]
        self.assertTrue(len(profile_evs) > 0)
        self.assertIn("cleared", profile_evs[0][1].lower())
        self.assertEqual(brain.profile_store.get_all(), [])


if __name__ == "__main__":
    unittest.main()
