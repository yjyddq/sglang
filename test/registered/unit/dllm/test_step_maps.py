"""Unit tests for the public dLLM step-map contract."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cpu_ci(est_time=2, suite="base-b-test-cpu")

import unittest

import torch

from sglang.srt.dllm.metadata import normalize_step_map_chunk


class TestDllmStepMaps(unittest.TestCase):

    def test_accepts_positive_one_based_tensor(self):
        self.assertEqual(
            normalize_step_map_chunk(torch.tensor([1, 3, 2]), 3, "request-0"),
            [1, 3, 2],
        )

    def test_rejects_token_count_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "2 steps and 3 tokens"):
            normalize_step_map_chunk([1, 2], 3, "request-0")

    def test_rejects_zero_based_step(self):
        with self.assertRaisesRegex(RuntimeError, "positive 1-based"):
            normalize_step_map_chunk([0], 1, "request-0")

    def test_rejects_boolean_step(self):
        with self.assertRaisesRegex(RuntimeError, "boolean"):
            normalize_step_map_chunk([True], 1, "request-0")

    def test_rejects_non_integer_step(self):
        with self.assertRaisesRegex(RuntimeError, "non-integer"):
            normalize_step_map_chunk([1.5], 1, "request-0")


if __name__ == "__main__":
    unittest.main()
