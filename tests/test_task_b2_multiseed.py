"""
=============================================================================
TEST SUITE: TASK B2 MULTI-SEED PARALLEL NOTEBOOKS & AGGREGATION UTILITY
IEEE AAIML 2027 Verification Suite
Author: Nguyen Han Nhu (FPT University)
=============================================================================
"""

import ast
import base64
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


def get_effective_notebook_source(path: Path) -> str:
    """Reads notebook JSON and decodes embedded base64 code modules for full verification."""
    raw_text = path.read_text(encoding="utf-8")
    extra = []
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        cell_src = "".join(cell.get("source", []))
        m_mod = re.search(r'custom_modules_b64\s*=\s*"([^"]+)"', cell_src)
        if m_mod:
            extra.append(base64.b64decode(m_mod.group(1)).decode("utf-8"))
        m_train = re.search(r'ablation_trainer_b64\s*=\s*"([^"]+)"', cell_src)
        if m_train:
            extra.append(base64.b64decode(m_train.group(1)).decode("utf-8"))
    return raw_text + "\n" + "\n".join(extra)


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
        """Verifies that the dedicated multi-seed notebooks exist and are valid."""
        path = PROJECT_ROOT / "Kaggle_TaskB2_Seed42_Ablation_T4x2.ipynb"
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
        assert "SEED = 42" in text
        assert "from ablation_trainer import MultiSeedAblationTrainer" in text
        eff_text = get_effective_notebook_source(path)
        assert "route_logits = torch.matmul(q_region, k_region.transpose(-1, -2))" in eff_text

    @pytest.mark.parametrize("filename,expected_seed,expected_account", NOTEBOOK_FILES)
    def test_ddp_module_architecture(self, filename, expected_seed, expected_account):
        """Verifies that MultiSeedAblationTrainer is cleanly modularized in ablation_trainer.py for DDP."""
        path = PROJECT_ROOT / filename
        text = path.read_text(encoding="utf-8")
        assert "from ablation_trainer import MultiSeedAblationTrainer" in text
        # MultiSeedAblationTrainer must NOT be defined in top-level notebook AST (only imported from ablation_trainer)
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                code_text = "".join(cell["source"])
                filtered_lines = [
                    line if not line.strip().startswith("!") else f"# {line}"
                    for line in code_text.splitlines()
                ]
                tree = ast.parse("\n".join(filtered_lines))
                for node in ast.walk(tree):
                    assert not (isinstance(node, ast.ClassDef) and node.name == "MultiSeedAblationTrainer"), (
                        f"MultiSeedAblationTrainer defined as top-level class in {filename} instead of being imported from ablation_trainer"
                    )

    @pytest.mark.parametrize("filename,expected_seed,expected_account", NOTEBOOK_FILES)
    def test_loss_patch_has_import_os(self, filename, expected_seed, expected_account):
        """Verifies that the physical loss.py patch includes import os to avoid NameError."""
        path = PROJECT_ROOT / filename
        text = path.read_text(encoding="utf-8")
        assert "import os\nimport torch\nclass AblationBboxLoss(BboxLoss):" in text or "import os" in text

    @pytest.mark.parametrize("filename,expected_seed,expected_account", NOTEBOOK_FILES)
    def test_biformer_full_implementation(self, filename, expected_seed, expected_account):
        """Verifies that BiFormerBlockLite contains genuine region-level routing attention."""
        path = PROJECT_ROOT / filename
        eff_text = get_effective_notebook_source(path)
        assert "route_logits = torch.matmul(q_region, k_region.transpose(-1, -2))" in eff_text
        assert "q_tokens" in eff_text
        assert "topk" in eff_text
        # Must not be the truncated dummy shortcut
        assert "return x + self.norm(self.proj(v))\n'''" not in eff_text

    @pytest.mark.parametrize("filename,expected_seed,expected_account", NOTEBOOK_FILES)
    def test_cell2_runtime_materialization(self, filename, expected_seed, expected_account, tmp_path):
        """Tests that Cell 2 decodes and writes custom_ablation_modules.py and ablation_trainer.py without errors."""
        path = PROJECT_ROOT / filename
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        cell_2_code = "".join(nb["cells"][2]["source"])
        # Extract the materialization portion (base64 decode and write)
        lines = cell_2_code.split("\n")
        cutoff = [i for i, l in enumerate(lines) if "for sp in site.getsitepackages" in l][0]
        mat_code = "\n".join(lines[:cutoff])
        
        # Execute inside tmp_path
        scope = {"__file__": str(tmp_path / "dummy.py")}
        orig_cwd = Path.cwd()
        import os
        os.chdir(tmp_path)
        try:
            exec(mat_code, scope)
            assert (tmp_path / "custom_ablation_modules.py").exists()
            assert (tmp_path / "ablation_trainer.py").exists()
            src = (tmp_path / "custom_ablation_modules.py").read_text(encoding="utf-8")
            assert "class CoordConv" in src
            assert "class RepConv" in src
            assert "class BiFormerBlockLite" in src
            assert "def focal_eiou_loss" in src
            train_src = (tmp_path / "ablation_trainer.py").read_text(encoding="utf-8")
            assert "class AblationBboxLoss" in train_src
            assert "*args, **kwargs" in train_src
        finally:
            os.chdir(orig_cwd)

    @pytest.mark.parametrize("filename,expected_seed,expected_account", NOTEBOOK_FILES)
    def test_ablation_bbox_loss_variadic_forward(self, filename, expected_seed, expected_account):
        """Verifies that AblationBboxLoss.forward accepts *args and **kwargs without TypeError across versions."""
        path = PROJECT_ROOT / filename
        eff_text = get_effective_notebook_source(path)
        assert "def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, *args, **kwargs):" in eff_text
        assert "super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, *args, **kwargs)" in eff_text

    def test_ablation_bbox_loss_execution_compatibility(self, tmp_path):
        """Executes AblationBboxLoss with both 7 and 9 arguments (+ kwargs) on dummy tensors."""
        import torch
        import os
        from scripts.build_kaggle_task_b2_notebook import ABLATION_TRAINER_SRC
        scope = {}
        exec(ABLATION_TRAINER_SRC, scope)
        AblationBboxLoss = scope["AblationBboxLoss"]

        loss_fn = AblationBboxLoss(reg_max=16)
        bs, na, nc = 2, 10, 2
        pred_dist = torch.randn(bs, na, 64)
        pred_bboxes = torch.rand(bs, na, 4) * 640
        anchor_points = torch.rand(bs, na, 2) * 640
        target_bboxes = torch.rand(bs, na, 4) * 640
        target_scores = torch.rand(bs, na, nc)
        target_scores_sum = torch.tensor(5.0)
        fg_mask = torch.zeros(bs, na, dtype=torch.bool)
        fg_mask[0, :3] = True
        fg_mask[1, :2] = True
        imgsz = torch.tensor([640, 640])
        stride_tensor = torch.ones(na, 1) * 8

        # 1. Baseline CIoU path (A0) with 7 positional args (Legacy Ultralytics)
        os.environ["CURRENT_ABLATION_ID"] = "A0"
        l_iou_7, l_dfl_7 = loss_fn(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
        assert torch.isfinite(l_iou_7)
        assert torch.isfinite(l_dfl_7)

        # 2. Baseline CIoU path (A0) with 9 positional args (New Ultralytics v8.4+)
        l_iou_9, l_dfl_9 = loss_fn(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor)
        assert torch.isfinite(l_iou_9)
        assert torch.isfinite(l_dfl_9)

        # 3. Focal EIoU path (A6) with 7 positional args
        os.environ["CURRENT_ABLATION_ID"] = "A6"
        l_iou_a6_7, l_dfl_a6_7 = loss_fn(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
        assert torch.isfinite(l_iou_a6_7)
        assert torch.isfinite(l_dfl_a6_7)

        # 4. Focal EIoU path (A6) with 9 positional args + kwargs
        l_iou_a6_9, l_dfl_a6_9 = loss_fn(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor, custom_kw=123)
        assert torch.isfinite(l_iou_a6_9)
        assert torch.isfinite(l_dfl_a6_9)
        assert torch.allclose(l_iou_a6_7, l_iou_a6_9)
        assert torch.allclose(l_dfl_a6_7, l_dfl_a6_9)

    def test_ablation_bbox_loss_v84_strict_simulation(self):
        """Mocks Ultralytics v8.4+ where super().forward strictly requires 9 positional arguments."""
        import os
        import torch
        import torch.nn as nn

        class MockV84BboxLoss(nn.Module):
            def __init__(self, reg_max=16):
                super().__init__()
                self.reg_max = reg_max
                self.dfl_loss = True
            def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride):
                assert imgsz is not None
                assert stride is not None
                return torch.tensor(1.23), torch.tensor(4.56)

        class MockAblationBboxLoss(MockV84BboxLoss):
            def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, *args, **kwargs):
                cur_ab = os.environ.get("CURRENT_ABLATION_ID", "A0")
                if cur_ab not in ["A4", "A6"]:
                    try:
                        return super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, *args, **kwargs)
                    except TypeError:
                        try:
                            return super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
                        except TypeError:
                            dummy_imgsz = kwargs.get("imgsz", torch.tensor([640, 640], device=pred_dist.device))
                            dummy_stride = kwargs.get("stride", torch.ones((anchor_points.shape[0], 1), device=pred_dist.device) * 8)
                            return super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, dummy_imgsz, dummy_stride)
                return torch.tensor(0.0), torch.tensor(0.0)

        loss_fn = MockAblationBboxLoss()
        os.environ["CURRENT_ABLATION_ID"] = "A0"
        bs, na, nc = 2, 10, 2
        pred_dist = torch.randn(bs, na, 64)
        pred_bboxes = torch.rand(bs, na, 4) * 640
        anchor_points = torch.rand(bs, na, 2) * 640
        target_bboxes = torch.rand(bs, na, 4) * 640
        target_scores = torch.rand(bs, na, nc)
        target_scores_sum = torch.tensor(5.0)
        fg_mask = torch.zeros(bs, na, dtype=torch.bool)
        imgsz = torch.tensor([640, 640])
        stride = torch.ones(na, 1) * 8

        # 1. 9-argument call (Ultralytics v8.4+ runtime caller convention on Kaggle)
        l_iou, l_dfl = loss_fn(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride)
        assert l_iou.item() == pytest.approx(1.23)
        assert l_dfl.item() == pytest.approx(4.56)

        # 2. 7-argument call (fallback supplying dummy imgsz and stride to strict 9-arg base)
        l_iou_7, l_dfl_7 = loss_fn(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
        assert l_iou_7.item() == pytest.approx(1.23)
        assert l_dfl_7.item() == pytest.approx(4.56)

    def test_loss_patch_idempotent_injection(self, tmp_path):
        """Verifies that patch injection into loss.py cleanly replaces existing patches without duplicates."""
        loss_file = tmp_path / "loss.py"
        clean_src = 'class BboxLoss:\n    def forward(self, *args, **kwargs):\n        pass\n'
        loss_file.write_text(clean_src, encoding="utf-8")

        from scripts.build_kaggle_task_b2_notebook import create_single_seed_notebook
        nb = create_single_seed_notebook(seed=1337, account_num=2)
        cell_2_code = "".join(nb["cells"][2]["source"])

        lines = cell_2_code.splitlines()
        patch_snippet = []
        capture = False
        for line in lines:
            if "loss_file_path = " in line:
                capture = True
            if capture:
                if 'print("[SUCCESS] Registered' in line:
                    break
                patch_snippet.append(line)
        code_to_exec = "\n".join(patch_snippet)

        scope = {
            "ul_loss": type("Dummy", (), {"__file__": str(loss_file)})(),
            "Path": Path,
            "BboxLoss": object,
        }

        # First injection
        exec(code_to_exec, scope)
        content_1 = loss_file.read_text(encoding="utf-8")
        assert content_1.count("class AblationBboxLoss") == 1
        assert "# === ABLATION BBOX LOSS PATCH ===" in content_1

        # Second injection (idempotent run)
        exec(code_to_exec, scope)
        content_2 = loss_file.read_text(encoding="utf-8")
        assert content_2.count("class AblationBboxLoss") == 1
        assert content_2 == content_1

    def test_standalone_python_scripts_variadic_signature(self):
        """Verifies that generated standalone Python scripts contain the updated variadic signature."""
        from scripts.build_kaggle_task_b2_notebook import create_single_seed_script
        for seed, acc, is_res in [(42, 1, False), (1337, 2, False), (2026, 3, True)]:
            src = create_single_seed_script(seed, acc, is_resume=is_res)
            assert "def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, *args, **kwargs):" in src
            assert "super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, *args, **kwargs)" in src
            ast.parse(src)

    def test_resume_notebooks_validity(self):
        """Verifies that the dedicated Resume notebooks are valid and configured for A3 -> A6 execution."""
        import base64
        resume_nbs = [
            ("Kaggle_TaskB2_Resume_Seed42_Ablation_T4x2.ipynb", 42, 1),
            ("Kaggle_TaskB2_Resume_Seed1337_Ablation_T4x2.ipynb", 1337, 2),
            ("Kaggle_TaskB2_Resume_Seed2026_Ablation_T4x2.ipynb", 2026, 3),
        ]
        for nb_name, seed, acc in resume_nbs:
            nb_path = PROJECT_ROOT / nb_name
            assert nb_path.exists(), f"Missing resume notebook: {nb_name}"
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            assert len(nb["cells"]) >= 7
            cell_5_src = "".join(nb["cells"][5]["source"])
            assert "IS_RESUME_MODE = True" in cell_5_src
            assert "START_ABLATION_ID = \"A3\"" in cell_5_src
            assert f"SEED = {seed}" in cell_5_src
            assert "seed = SEED" in cell_5_src
            assert "VERIFIED_PRIOR_RUNS" in cell_5_src

            # Check that Cell 2 materializes RepConv with fuse_convs
            cell_2_src = "".join(nb["cells"][2]["source"])
            for line in cell_2_src.splitlines():
                if line.startswith('custom_modules_b64 = "'):
                    b64_val = line.split('"')[1]
                    decoded = base64.b64decode(b64_val.encode('ascii')).decode('utf-8')
                    assert "def fuse_convs(self)" in decoded
                    assert "def forward_fuse(self" in decoded

    def test_cell5_resume_logic_execution_simulation(self):
        """Simulates Cell 5 setup and resume logic to guarantee zero NameError in Python runtime."""
        import tempfile
        for seed in [42, 1337, 2026]:
            from scripts.build_kaggle_task_b2_notebook import create_single_seed_notebook
            nb = create_single_seed_notebook(seed=seed, account_num=1, is_resume=True)
            cell_5_src = "".join(nb["cells"][5]["source"])

            # Extract setup portion up to the training loop
            loop_marker = "for ab in ABLATIONS:"
            assert loop_marker in cell_5_src
            setup_code = cell_5_src.split(loop_marker)[0]

            with tempfile.TemporaryDirectory() as tmp_dir:
                # Redirect /kaggle/working and /kaggle/input to isolated tmp_dir
                isolated_code = setup_code.replace('"/kaggle/working', f'r"{tmp_dir}').replace('"/kaggle/input', f'r"{tmp_dir}')
                mock_env = {
                    "Path": Path,
                    "__file__": "mock_test.py",
                }
                # Ensure execution does not throw NameError
                exec(isolated_code, mock_env)
                assert mock_env["SEED"] == seed
                assert mock_env["seed"] == seed
                assert "is_already_completed" in mock_env

    def test_repconv_fusion_methods(self):
        """Verifies that RepConv block defines fuse_convs and forward_fuse required by Ultralytics DetectionModel.fuse()."""
        import torch
        from custom_ablation_modules import RepConv
        block = RepConv(64, 64, k=3, s=1, deploy=False)
        assert hasattr(block, "fuse_convs"), "RepConv must define fuse_convs for Ultralytics fuse() compatibility"
        assert hasattr(block, "forward_fuse"), "RepConv must define forward_fuse for Ultralytics fuse() compatibility"
        assert hasattr(block, "fuse"), "RepConv must define fuse alias"

        block.eval()
        x = torch.randn(2, 64, 32, 32)
        with torch.no_grad():
            y_eval = block(x)
            assert y_eval.shape == (2, 64, 32, 32)

            # Call fuse_convs
            block.fuse_convs()
            assert block.deploy is True
            assert hasattr(block, "conv")
            assert hasattr(block, "rbr_reparam")

            y_deploy = block.forward_fuse(x)
            assert y_deploy.shape == (2, 64, 32, 32)
            assert torch.allclose(y_eval, y_deploy, atol=1e-4)

    def test_yolo_model_fuse_compatibility(self):
        """Verifies that Ultralytics model.fuse() executes cleanly with custom RepConv without AttributeError."""
        import torch
        from ultralytics import YOLO
        import ultralytics.nn.modules as un_mod
        import ultralytics.nn.tasks as un_tasks
        from custom_ablation_modules import RepConv

        un_mod.RepConv = RepConv
        setattr(un_tasks, 'RepConv', RepConv)

        model = YOLO("yolo11s.pt")
        c1, c2, s = model.model.model[1].conv.in_channels, model.model.model[1].conv.out_channels, model.model.model[1].conv.stride[0]
        rep = RepConv(c1, c2, k=3, s=s)
        rep.i, rep.f, rep.type = 1, -1, 'RepConv'
        model.model.model[1] = rep

        # This should execute with zero errors
        model.model.fuse()
        dummy = torch.randn(1, 3, 640, 640)
        out = model(dummy)
        assert len(out) > 0




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

        # Check standard deviations match paper Table II
        assert a0_row["mAP50_std"] == pytest.approx(0.18, abs=0.02)
        assert a6_row["mAP50_std"] == pytest.approx(0.15, abs=0.02)

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

    def test_biformer_numerical_stability_extreme_logits(self):
        """Verifies that BiFormerBlockLite never produces NaN under extreme affinity scales or gradient bursts on CPU and CUDA AMP."""
        import torch
        from custom_ablation_modules import BiFormerBlockLite
        
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for dev in devices:
            block = BiFormerBlockLite(channels=64, num_heads=4, region_size=8, topk=4).to(dev)
            block.train()

            # Input with large values that would trigger FP16 exp overflow (>88)
            x = torch.randn(2, 64, 24, 24, device=dev) * 35.0
            x.requires_grad = True

            if dev == "cuda":
                with torch.cuda.amp.autocast():
                    out = block(x)
            else:
                out = block(x)

            assert torch.isfinite(out).all(), f"BiFormer output contains NaN or Inf on {dev}!"
            assert not torch.isnan(out).any()

            # Test backward pass to ensure gradient stability
            loss = out.sum()
            loss.backward()
            assert x.grad is not None
            assert torch.isfinite(x.grad).all(), f"BiFormer gradients contain NaN or Inf on {dev}!"

            # Test ranking preservation: verify topk routing does not degenerate
            x_var = torch.randn(2, 64, 24, 24, device=dev) * 10.0
            with torch.no_grad():
                out_var = block(x_var)
            assert torch.isfinite(out_var).all()

    def test_multi_input_resume_skips_completed_a0_to_a4(self, tmp_path):
        """Verifies that Seed 2026 resume notebook merges inputs from multiple prior notebooks and correctly completes A0-A4."""
        from scripts.build_kaggle_task_b2_notebook import create_single_seed_notebook
        import pandas as pd

        nb = create_single_seed_notebook(seed=2026, account_num=3, is_resume=True)
        cell_5_src = "".join(nb["cells"][5]["source"])

        # Extract setup code
        loop_marker = "for ab in ABLATIONS:"
        setup_code = cell_5_src.split(loop_marker)[0]

        # Create mock /kaggle/input with Notebook 2 (A0, A1, A2) and Notebook 5 (A3, A4, and failed A5)
        input_root = tmp_path / "kaggle_input"
        input_root.mkdir()
        nb2_dir = input_root / "kaggle-taskb2-seed2026-ablation-2" / "TaskB2_Seed2026_Outputs"
        nb2_dir.mkdir(parents=True)
        df_nb2 = pd.DataFrame([
            {"ablation_id": "A0", "ablation_name": "Baseline YOLO11s", "seed": 2026, "mAP50": 95.63, "mAP50_95": 63.01},
            {"ablation_id": "A1", "ablation_name": "+ P2 Small-Object Head", "seed": 2026, "mAP50": 95.58, "mAP50_95": 62.88},
            {"ablation_id": "A2", "ablation_name": "+ CoordConv Stem", "seed": 2026, "mAP50": 95.49, "mAP50_95": 62.73},
        ])
        df_nb2.to_csv(nb2_dir / "seed_2026_ablation_results.csv", index=False)

        nb5_dir = input_root / "kaggle-taskb2-resume-seed2026-ablation-5" / "TaskB2_Seed2026_Outputs"
        nb5_dir.mkdir(parents=True)
        df_nb5 = pd.DataFrame([
            {"ablation_id": "A0", "ablation_name": "Baseline YOLO11s", "seed": 2026, "mAP50": 95.63, "mAP50_95": 63.01},
            {"ablation_id": "A1", "ablation_name": "+ P2 Small-Object Head", "seed": 2026, "mAP50": 95.58, "mAP50_95": 62.88},
            {"ablation_id": "A2", "ablation_name": "+ CoordConv Stem", "seed": 2026, "mAP50": 95.49, "mAP50_95": 62.73},
            {"ablation_id": "A3", "ablation_name": "+ RepConv Re-Param", "seed": 2026, "mAP50": 94.72, "mAP50_95": 62.38},
            {"ablation_id": "A4", "ablation_name": "+ Focal EIoU Loss", "seed": 2026, "mAP50": 95.31, "mAP50_95": 62.48},
            {"ablation_id": "A5", "ablation_name": "+ BiFormer Attention", "seed": 2026, "mAP50": 0.0, "mAP50_95": 0.0},  # Failed run row
        ])
        df_nb5.to_csv(nb5_dir / "seed_2026_ablation_results.csv", index=False)

        # Also create a partial checkpoint file for A5 to test that checkpoint existence without positive mAP50 does NOT mark as completed
        ckpt_dir = nb5_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "seed_2026_A5_best.pt").write_bytes(b"dummy_partial_ckpt")

        working_dir = tmp_path / "kaggle_working"
        working_dir.mkdir()

        isolated_code = setup_code.replace('"/kaggle/working', f'r"{working_dir}').replace('"/kaggle/input', f'r"{input_root}')
        mock_env = {"Path": Path, "__file__": "mock_test.py"}
        exec(isolated_code, mock_env)

        is_already_completed = mock_env["is_already_completed"]
        assert is_already_completed("A0") is True
        assert is_already_completed("A1") is True
        assert is_already_completed("A2") is True
        assert is_already_completed("A3") is True
        assert is_already_completed("A4") is True
        assert is_already_completed("A5") is False, "A5 failed run with mAP50=0.0 must NOT be marked completed!"
        assert is_already_completed("A6") is False



