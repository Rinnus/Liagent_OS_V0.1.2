import unittest
from unittest.mock import patch
import json

from liagent.engine.provider_registry import (
    get_provider_profile,
    infer_api_model_family,
    infer_api_provider,
    infer_api_tool_protocol,
    list_provider_presets,
)


class ProviderRegistryTests(unittest.TestCase):
    def test_infer_moonshot_provider_from_direct_endpoint(self):
        provider = infer_api_provider("kimi-k2.5", "https://api.moonshot.cn/v1")
        self.assertEqual(provider, "moonshot")

    def test_infer_moonshot_provider_from_openrouter_vendor_model(self):
        provider = infer_api_provider("moonshotai/kimi-k2.5", "https://openrouter.ai/api/v1")
        self.assertEqual(provider, "moonshot")

    def test_infer_model_family_for_kimi_defaults_to_openai_family(self):
        family = infer_api_model_family("moonshotai/kimi-k2.5", "https://openrouter.ai/api/v1")
        self.assertEqual(family, "openai")

    def test_provider_presets_include_moonshot_and_openrouter(self):
        presets = list_provider_presets(include_custom=False)
        ids = {str(item.get("id", "")) for item in presets}
        self.assertIn("moonshot", ids)
        self.assertIn("openrouter", ids)

    def test_infer_tool_protocol_defaults_to_openai_function_for_cloud_api(self):
        protocol = infer_api_tool_protocol(
            "moonshotai/kimi-k2.5",
            "https://openrouter.ai/api/v1",
        )
        self.assertEqual(protocol, "openai_function")

    def test_moonshot_profile_declares_fixed_temperature_policy(self):
        profile = get_provider_profile("moonshot")
        self.assertEqual(profile.temperature_policy, "fixed")
        self.assertEqual(float(profile.temperature_fixed or 0.0), 1.0)

    def test_env_provider_extend_inherits_fixed_temperature_without_keyerror(self):
        payload = {
            "kimi_gateway": {
                "extends": "moonshot",
                "api_base_url": "https://example.com/v1",
                "api_model": "kimi-k2.5",
            }
        }
        with patch.dict(
            "os.environ",
            {"LIAGENT_API_PROVIDER_REGISTRY_JSON": json.dumps(payload)},
            clear=False,
        ):
            profile = get_provider_profile("kimi_gateway")
        self.assertEqual(profile.temperature_policy, "fixed")
        self.assertEqual(float(profile.temperature_fixed or 0.0), 1.0)
