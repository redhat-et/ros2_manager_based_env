"""Pure-Python contract tests: python -m unittest <package>.test_contract."""
import unittest
import numpy as np
from .contract import (ChunkBuffer, decode_chunk, joint_targets, closure_from_fingers,
                       CONTROL_HZ, PHYSICS_HZ, DECIMATION)


class ContractTests(unittest.TestCase):
    def test_rates(self):
        self.assertEqual(PHYSICS_HZ / DECIMATION, CONTROL_HZ)

    def test_no_chunk_is_idle(self):
        self.assertIsNone(ChunkBuffer().pop())

    def test_once_per_tick_then_idle(self):
        b = ChunkBuffer()
        a = np.arange(15 * 8, dtype=np.float32).reshape(15, 8)
        b.replace(a)
        for row in a:
            np.testing.assert_array_equal(b.pop(), row)
        self.assertEqual(b.remaining, 0)
        self.assertIsNone(b.pop())
        self.assertIsNone(b.pop())

    def test_empty_cancels_motion(self):
        b = ChunkBuffer()
        b.replace(np.ones((15, 8)))
        b.replace(decode_chunk([]))
        self.assertIsNone(b.pop())

    def test_new_chunk_replaces_remainder(self):
        b = ChunkBuffer()
        b.replace(np.ones((15, 8)))
        b.pop()
        b.replace(np.zeros((1, 8)))
        np.testing.assert_array_equal(b.pop(), np.zeros(8))
        self.assertIsNone(b.pop())

    def test_bad_chunks(self):
        for data, shape, offset in [([1] * 7, (), 0), ([1] * 128, (), 0),
                                    ([1] * 8, (2, 8), 0), ([1] * 8, (1, 7), 0),
                                    ([float("nan")] * 8, (), 0), ([1] * 8, (), 1)]:
            with self.subTest(shape=shape, size=len(data)):
                with self.assertRaises(ValueError):
                    decode_chunk(data, shape, offset)

    def test_source_not_mutated(self):
        b = ChunkBuffer()
        a = np.ones((1, 8))
        b.replace(a)
        a[:] = 0
        np.testing.assert_array_equal(b.pop(), np.ones(8))

    def test_gripper_closure(self):
        np.testing.assert_allclose(closure_from_fingers([[0.04, 0.04], [0, 0], [0.02, 0.02]]), [0, 1, 0.5])

    def test_velocity_is_not_position(self):
        q = np.array([1.0] * 7)
        np.testing.assert_allclose(joint_targets(q, [0.5] * 7, [-3] * 7, [3] * 7), [1.1] * 7)

    def test_joint_limit_clip(self):
        np.testing.assert_allclose(joint_targets([0.95] * 7, [10] * 7, [-1] * 7, [1] * 7), [1] * 7)


if __name__ == "__main__":
    unittest.main()
