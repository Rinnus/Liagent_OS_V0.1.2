import unittest


class ComputeConfidenceTests(unittest.TestCase):
    def test_high_confidence(self):
        from liagent.agent.response_guard import compute_confidence_label
        label, note = compute_confidence_label(
            evidence_list=[{"tool": "web_search"}, {"tool": "stock"}],
            quality_issues=[],
            experience_score=1.5,
        )
        self.assertEqual(label, "high")
        self.assertIsNone(note)

    def test_low_confidence_no_sources(self):
        from liagent.agent.response_guard import compute_confidence_label
        label, note = compute_confidence_label(
            evidence_list=[],
            quality_issues=[],
            experience_score=0.5,
        )
        self.assertEqual(label, "low")
        self.assertIsNotNone(note)

    def test_low_confidence_with_issues(self):
        from liagent.agent.response_guard import compute_confidence_label
        label, note = compute_confidence_label(
            evidence_list=[{"tool": "web_search"}],
            quality_issues=["unsourced_data"],
            experience_score=1.0,
        )
        self.assertEqual(label, "low")

    def test_medium_confidence(self):
        from liagent.agent.response_guard import compute_confidence_label
        label, note = compute_confidence_label(
            evidence_list=[{"tool": "web_search"}],
            quality_issues=[],
            experience_score=0.8,
        )
        self.assertEqual(label, "medium")


if __name__ == "__main__":
    unittest.main()
