"""Tests for test threshold-sweep table parsing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.plot_test_threshold_sweep import (
    output_path_for_metric,
    plot_threshold_grid,
    read_threshold_table,
)


class ReadThresholdTableTest(unittest.TestCase):
    """Verify the W&B table parser preserves the relevant score series."""

    def test_reads_tss_and_hss_by_probability_threshold(self) -> None:
        """A valid W&B threshold table is converted to numeric rows."""
        payload = {
            "columns": ["threshold", "TSS", "HSS", "CSS", "F1_macro"],
            "data": [
                [0.01, 0.15, 0.12, 0.13, 0.55],
                [0.50, 0.41, 0.43, 0.42, 0.72],
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            table_path = Path(directory) / "threshold_df.table.json"
            table_path.write_text(json.dumps(payload))

            rows = read_threshold_table(table_path)

        self.assertEqual(rows, [(0.01, 0.15, 0.12), (0.50, 0.41, 0.43)])

    def test_uses_a_condition_specific_output_stem(self) -> None:
        """Different forecast configurations do not overwrite each other's plots."""
        output = output_path_for_metric(Path("results/plots"), "test_2h_m_1h", "TSS")

        self.assertEqual(
            output,
            Path("results/plots/test_2h_m_1h_tss_threshold_sweep.png"),
        )

    def test_writes_a_two_condition_threshold_grid(self) -> None:
        """A four-panel TSS/HSS grid is written for two forecast conditions."""
        curves = {
            "24-hour forecast | 8-hour sampling": {
                "Surya": [(0.01, 0.1, 0.1), (0.5, 0.4, 0.4)],
                "AlexNet": [(0.01, 0.2, 0.2), (0.5, 0.3, 0.3)],
            },
            "2-hour forecast | 1-hour sampling": {
                "Surya": [(0.01, 0.1, 0.1), (0.5, 0.2, 0.2)],
                "AlexNet": [(0.01, 0.2, 0.2), (0.5, 0.5, 0.5)],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "grid.png"
            plot_threshold_grid(curves, output)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
