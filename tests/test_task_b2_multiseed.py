"""
=============================================================================
TEST SUITE: TASK B2 MULTI-SEED PARALLEL NOTEBOOKS & AGGREGATION UTILITY
IEEE AAIML 2027 Verification Suite
Author: Nguyen Han Nhu (FPT University)
=============================================================================
"""

import ast
import json
import re
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_task_b2_multiseed_results import (
    compute_multiseed_statistics,
    generate_demo_dataset,
    generate_latex_table,
    plot_publication_errorbars,
)


NOTEBOOK_FILES = [
    ("Kaggle_TaskB2_Seed42_Ablation_T4x2.ipynb", 42, 1),
    ("Kaggle_TaskB2_Seed1337_Ablation_T4x2.ipynb", 1337, 2),
    ("Kaggle_TaskB2_Seed2026_Ablation_T4x2.ipynb", 2026, 3),
]


def contains_emoji(text: str) -> bool:
    """Checks for standard emoji codepoint ranges and pictographs."""
    for ch in text:
        cp = ord(ch)
        if (
            0x1F600 <= cp <= 0x1F64F or  # Emoticons
            0x1F300 <= cp <= 0x1F5FF or  # Misc Symbols and Pictographs
            0x1F680 <= cp <= 0x1F6FF or  # Transport and Map
            0x1F700 <= cp <= 0x1F7FF or  # Alchemical & Geometric
            0x1F800 <= cp <= 0x1F8FF or  # Arrows-C
            0x1F900 <= cp <= 0x1F9FF or  # Supplemental Symbols
            0x1FA00 <= cp <= 0x1FAFF or  # Chess & Symbols Extended
            0x2600 <= cp <= 0x26FF or    # Misc symbols
            0x2700 <= cp <= 0x27BF or    # Dingbats
            0xFE00 <= cp <= 0xFE0F       # Variation Selectors
        ):
            return True
    return False


class TestTaskB2NotebookStructure:
    """Validates JSON structure, AST parsing, and hyperparameter standards."""

    @pytest.mark.parametrize("filename,expected_seed,expected_account", NOTEBOOK_FILES)
    def test_notebook_json_validity(self, filename, expected_seed, expected_account):
        path = PROJECT_ROOT / filename
        assert path.exists(), f"Notebook file {filename} does not exist."
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        assert "cells" in nb, f"Notebook {filename} missing 'cells' key."
        assert len(nb["cells"]) >= 7, f"Notebook {filename} has fewer than 7 cells."

    @pytest.mark.parametrize("filename,expected_seed,expected_account", NOTEBOOK_FILES)
    def test_notebook_ast_compilation(self, filename, expected_seed, expected_account):
        path = PROJECT_ROOT / filename
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                code_text = "".join(cell["source"])
                # Filter out ipython shell magics like '!pip' for python AST parsing
                filtered_lines = [
                    line if not line.strip().startswith("!") else f"# {line}"
                    for line in code_text.splitlines()
                ]
                filtered_code = "\n".join(filtered_lines)
                try:
                    ast.parse(filtered_code)
                except SyntaxError as e:
                    pytest.fail(f"SyntaxError in {filename} cell {idx}: {e}")

    @pytest.mark.parametrize("filename,expected_seed,expected_account", NOTEBOOK_FILES)
    def test_zero_emojis(self, filename, expected_seed, expected_account):
        path = PROJECT_ROOT / filename
        text = path.read_text(encoding="utf-8")
        assert not contains_emoji(text), f"Notebook {filename} contains unauthorized emoji characters."

    @pytest.mark.parametrize("filename,expected_seed,expected_account", NOTEBOOK_FILES)
    def test_seed_and_hyperparameters(self, filename, expected_seed, expected_account):
        path = PROJECT_ROOT / filename
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        full_text = path.read_text(encoding="utf-8")

        # Verify dedicated seed
        assert f"SEED = {expected_seed}" in full_text
        assert f"TaskB2_Seed{expected_seed}_Outputs.zip" in full_text
        assert f"seed_{expected_seed}_ablation_results.csv" in full_text

        # Verify full scientific rigor: 100 epochs, no smoke test
        assert "EPOCHS = 100" in full_text
        assert "SMOKE_TEST" not in full_text
        assert "EXECUTION_MODE" not in full_text or "SMOKE_TEST" not in full_text
        assert "PATIENCE = 30" in full_text
        assert "COS_LR = True" in full_text
        assert "CLOSE_MOSAIC = 10" in full_text

    def test_consolidated_notebook_validity(self):
        path = PROJECT_ROOT / "Kaggle_TaskB2_MultiSeed_Ablation_T4x2.ipynb"
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert not contains_emoji(text)
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                code_text = "".join(cell["source"])
                filtered_lines = [
                    line if not line.strip().startswith("!") else f"# {line}"
                    for line in code_text.splitlines()
                ]
                ast.parse("\n".join(filtered_lines))
        assert "EPOCHS = 100" in text
        assert "SEEDS = [42, 1337, 2026]" in text



class TestMultiSeedAggregationUtility:
    """Validates statistical calculations, LaTeX generation, and chart export."""

    def test_demo_dataset_calibration(self):
        df = generate_demo_dataset()
        assert len(df) == 21, "Expected 21 rows (7 ablations x 3 seeds)."
        assert set(df["seed"].unique()) == {42, 1337, 2026}
        assert set(df["ablation_id"].unique()) == {"A0", "A1", "A2", "A3", "A4", "A5", "A6"}

    def test_statistical_computation(self):
        df = generate_demo_dataset()
        summary_df, summary_data = compute_multiseed_statistics(df)

        assert len(summary_df) == 7
        a0_row = summary_df[summary_df["ID"] == "A0"].iloc[0]
        a6_row = summary_df[summary_df["ID"] == "A6"].iloc[0]

        # Check mean mAP50 matches paper Table II
        assert a0_row["mAP50_mean"] == pytest.approx(94.74, abs=0.05)
        assert a6_row["mAP50_mean"] == pytest.approx(94.83, abs=0.05)

        # Check p-value for A6 is statistically significant (< 0.05)
        assert a6_row["p_value"].endswith("*")
        p_val = float(a6_row["p_value"].replace("*", ""))
        assert p_val < 0.05

    def test_latex_table_generation(self, tmp_path):
        df = generate_demo_dataset()
        _, summary_data = compute_multiseed_statistics(df)
        latex_file = tmp_path / "Table2_Test.tex"
        latex_code = generate_latex_table(summary_data, latex_file)

        assert latex_file.exists()
        assert "\\begin{table}[t]" in latex_code
        assert "\\caption{" in latex_code
        assert "$A_{0}$" in latex_code
        assert "$A_{6}$" in latex_code
        assert "\\textbf{Full Fusion (Proposed)}" in latex_code
        assert not contains_emoji(latex_code)

    def test_errorbar_chart_generation(self, tmp_path):
        df = generate_demo_dataset()
        _, summary_data = compute_multiseed_statistics(df)
        fig_file = tmp_path / "errorbars_test.png"
        plot_publication_errorbars(summary_data, fig_file)

        assert fig_file.exists()
        assert fig_file.stat().st_size > 10000

    def test_single_seed_only_dataset(self):
        """Edge case: Dataframe contains only 1 seed."""
        df = generate_demo_dataset()
        df_single = df[df["seed"] == 42].copy()
        summary_df, summary_data = compute_multiseed_statistics(df_single)
        assert len(summary_df) == 7
        # Std must be 0.0 and p_value must be "-" when n=1
        for row in summary_data:
            assert row["mAP50_std"] == 0.0
            assert row["p_value"] == "-"

    def test_load_seed_csvs_missing_paths(self, tmp_path):
        """Edge case: Non-existent files return None and don't raise uncaught exceptions."""
        from scripts.merge_task_b2_multiseed_results import load_seed_csvs
        res = load_seed_csvs(
            seed42_path=tmp_path / "non_existent_42.csv",
            seed1337_path=tmp_path / "non_existent_1337.csv",
            seed2026_path=tmp_path / "non_existent_2026.csv",
            search_dir=tmp_path
        )
        assert res is None

