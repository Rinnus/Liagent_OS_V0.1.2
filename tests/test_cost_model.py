import os
import unittest
from unittest.mock import patch

from liagent.engine.cost_model import estimate_usage_cost_usd, resolve_pricing_rates


class CostModelTests(unittest.TestCase):
    def test_estimate_usage_cost_accounts_for_cached_tokens(self):
        cost = estimate_usage_cost_usd(
            provider="gemini",
            model="gemini-3.0-flash",
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cached_prompt_tokens=800_000,
            cache_write_tokens=0,
        )
        # Cached prompt tokens should materially reduce cost.
        self.assertLess(cost, 0.20)

    def test_provider_env_override(self):
        with patch.dict(
            os.environ,
            {
                "LIAGENT_COST_OPENAI_PROMPT_PER_MTOK": "1.0",
                "LIAGENT_COST_OPENAI_COMPLETION_PER_MTOK": "2.0",
            },
            clear=False,
        ):
            rates = resolve_pricing_rates("openai", "gpt-4o")
        self.assertEqual(rates.prompt_per_mtok, 1.0)
        self.assertEqual(rates.completion_per_mtok, 2.0)

