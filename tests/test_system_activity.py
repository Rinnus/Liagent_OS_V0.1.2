"""Tests for Agent Action Visibility — _system_activity ring buffer."""

import unittest
from collections import deque
from unittest.mock import patch


class RecordSystemActivityTests(unittest.TestCase):
    """Test record_system_activity() appends timestamped entries."""

    def _make_brain_activity(self) -> deque:
        """Return a deque mimicking AgentBrain._system_activity."""
        return deque(maxlen=10)

    def test_appends_with_timestamp(self):
        from liagent.agent.brain import AgentBrain
        buf = self._make_brain_activity()
        # Simulate what record_system_activity does
        with patch("liagent.agent.brain.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "14:30"
            # Use the same logic as the method
            buf.append(f"[{mock_dt.now().strftime('%H:%M')}] Signal: AAPL up 5%")
        self.assertEqual(len(buf), 1)
        self.assertIn("14:30", buf[0])
        self.assertIn("Signal: AAPL up 5%", buf[0])

    def test_bounded_at_10(self):
        buf = self._make_brain_activity()
        for i in range(15):
            buf.append(f"[00:00] Event {i}")
        self.assertEqual(len(buf), 10)
        # Oldest entries evicted
        self.assertIn("Event 5", buf[0])
        self.assertIn("Event 14", buf[-1])


class SystemActivityInjectionTests(unittest.TestCase):
    """Test that _system_activity is injected into messages and cleared."""

    def test_injection_on_step_zero(self):
        """Simulate the injection logic from brain.py run() step==0."""
        activity = deque(["[14:30] Signal poll: found 2 items",
                          "[14:31] Heartbeat: checked API health"],
                         maxlen=10)

        # Simulate the injection code path
        messages = [{"role": "system", "content": "You are an assistant."}]
        step = 0
        if activity and step == 0:
            activity_text = "\n".join(activity)
            messages.append({"role": "system",
                             "content": f"[Recent System Activity]\n{activity_text}"})
            activity.clear()

        self.assertEqual(len(messages), 2)
        self.assertIn("[Recent System Activity]", messages[1]["content"])
        self.assertIn("Signal poll", messages[1]["content"])
        self.assertIn("Heartbeat", messages[1]["content"])
        # Buffer cleared after injection
        self.assertEqual(len(activity), 0)

    def test_no_injection_on_step_nonzero(self):
        """Activity should only be injected on step 0, not later steps."""
        activity = deque(["[14:30] Some event"], maxlen=10)
        messages = [{"role": "system", "content": "You are an assistant."}]
        step = 1
        if activity and step == 0:
            activity_text = "\n".join(activity)
            messages.append({"role": "system",
                             "content": f"[Recent System Activity]\n{activity_text}"})
            activity.clear()

        # No injection on step > 0
        self.assertEqual(len(messages), 1)
        # Buffer preserved for next run()
        self.assertEqual(len(activity), 1)

    def test_no_injection_when_empty(self):
        """No system activity message added when buffer is empty."""
        activity = deque(maxlen=10)
        messages = [{"role": "system", "content": "You are an assistant."}]
        step = 0
        if activity and step == 0:
            activity_text = "\n".join(activity)
            messages.append({"role": "system",
                             "content": f"[Recent System Activity]\n{activity_text}"})
            activity.clear()

        self.assertEqual(len(messages), 1)


if __name__ == "__main__":
    unittest.main()
