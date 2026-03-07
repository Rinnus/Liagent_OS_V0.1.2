"""Pattern normalization, grouping via Union-Find, and GoalStore integration."""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .goal_store import GoalStore

_INTENT_MAP = {
    "stock_query": "price_check",
    "web_search": "info_search",
    "tool_use:web_search": "info_search",
    "tool_use:stock": "price_check",
    "topic_mention": "interest",
    "page_visit": "research",
}


def normalize_patterns(raw_patterns: list[dict]) -> list[dict]:
    normalized = []
    for p in raw_patterns:
        entities = _extract_entities(p["key"])
        intent = _INTENT_MAP.get(p["signal_type"], "general")
        normalized.append({**p, "entities": entities, "intent": intent})
    return normalized


def _extract_entities(key: str) -> list[str]:
    skip = {"stock", "web", "news", "tool", "search"}
    parts = key.split(":")
    entities = [p for p in parts if len(p) >= 2 and p.lower() not in skip]
    return entities if entities else [key]


class UnionFind:
    def __init__(self, n: int):
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


def _should_merge(a: dict, b: dict) -> bool:
    if a["domain"] != b["domain"]:
        return False
    ents_a = set(a.get("entities", []))
    ents_b = set(b.get("entities", []))
    if ents_a & ents_b:
        return True
    if a.get("intent") == b.get("intent"):
        return True
    return False


def _compute_group_key(members: list[dict]) -> str:
    domain = members[0]["domain"]
    all_entities = sorted({e for m in members for e in m.get("entities", [])})
    raw = f"{domain}:{','.join(all_entities)}"
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{domain}:{h}"


def _collect_entities(members: list[dict]) -> list[str]:
    return sorted({e for m in members for e in m.get("entities", [])})


def _collect_intents(members: list[dict]) -> list[str]:
    return sorted({m.get("intent", "general") for m in members})


def update_pattern_groups(patterns: list[dict], store: GoalStore) -> list[int]:
    if not patterns:
        return []

    uf = UnionFind(len(patterns))
    for i, j in combinations(range(len(patterns)), 2):
        if _should_merge(patterns[i], patterns[j]):
            uf.union(i, j)

    groups: dict[int, list[dict]] = {}
    for idx in range(len(patterns)):
        root = uf.find(idx)
        groups.setdefault(root, []).append(patterns[idx])

    new_ids = []
    for members in groups.values():
        group_key = _compute_group_key(members)
        existing = store.get_group_by_key(group_key)
        if existing:
            store.update_group_support(existing["id"], new_count=len(members))
        else:
            gid = store.create_group(
                group_key=group_key,
                domain=members[0]["domain"],
                entities=_collect_entities(members),
                intents=_collect_intents(members),
                support_count=len(members),
            )
            new_ids.append(gid)
    return new_ids
