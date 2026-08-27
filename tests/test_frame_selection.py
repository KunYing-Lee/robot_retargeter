from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from smpl_replay import select_frame_slice, slice_frame_aligned_arrays


class FrameSelectionTest(unittest.TestCase):
    def test_frame_range_is_inclusive_and_export_arrays_stay_aligned(self) -> None:
        frame_ids = select_frame_slice(10, start_frame=2, end_frame=8, stride=2)
        np.testing.assert_array_equal(
            frame_ids, np.array([2, 4, 6, 8], dtype=np.int32)
        )

        positions = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)
        states = np.arange(10, dtype=np.int64)
        selected = slice_frame_aligned_arrays(
            frame_ids,
            positions=positions,
            states=states,
            optional=None,
        )

        np.testing.assert_array_equal(selected["positions"], positions[frame_ids])
        np.testing.assert_array_equal(selected["states"], states[frame_ids])
        self.assertIsNone(selected["optional"])

    def test_frame_selection_rejects_misaligned_arrays(self) -> None:
        frame_ids = select_frame_slice(10, start_frame=7, end_frame=9, stride=1)
        with self.assertRaisesRegex(ValueError, "has 9 frames"):
            slice_frame_aligned_arrays(frame_ids, truncated=np.zeros((9, 2)))


if __name__ == "__main__":
    unittest.main()
