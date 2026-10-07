"""
=============================================================================
UNIT TESTS: BIPARTITE PERSON-HELMET ASSOCIATION & POSTER FALSE POSITIVE FILTER
IEEE AAIML 2027 Verification Suite (Task B3)
Author: Nguyen Han Nhu (FPT University)
=============================================================================
"""

import sys
from pathlib import Path

# Add project root to sys.path so tests can be run from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from rtsp_pipeline import compute_box_intersection_ratio, filter_poster_false_positives


class TestBipartiteSpatialFilter:
    """Test suite for Eq. 24 Bipartite Spatial Association Filter."""

    def test_compute_box_intersection_ratio_perfect_overlap(self):
        """Helmet is fully inside the person's upper torso."""
        # Person: [100, 100, 300, 500] (height=400, upper torso top 60% = 100 to 340)
        # Helmet: [150, 100, 250, 200] (inside upper torso)
        p_box = [100, 100, 300, 500]
        h_box = [150, 100, 250, 200]
        ratio = compute_box_intersection_ratio(h_box, p_box, use_upper_torso=True)
        assert ratio == pytest.approx(1.0, rel=1e-3), f"Expected 1.0, got {ratio}"

    def test_compute_box_intersection_ratio_zero_overlap(self):
        """Helmet is far away from the person."""
        p_box = [100, 100, 200, 400]
        h_box = [500, 500, 600, 600]
        ratio = compute_box_intersection_ratio(h_box, p_box)
        assert ratio == 0.0

    def test_compute_box_intersection_ratio_partial_overlap(self):
        """Helmet partially overlaps upper torso."""
        # Helmet area = 100 x 100 = 10000
        # Intersection = 50 x 100 = 5000 -> ratio = 0.50
        p_box = [100, 100, 200, 400]
        h_box = [150, 100, 250, 200]  # overlap x: [150, 200] (w=50), y: [100, 200] (h=100)
        ratio = compute_box_intersection_ratio(h_box, p_box)
        assert ratio == pytest.approx(0.50, rel=1e-2)

    def test_filter_poster_false_positive_rejection(self):
        """Isolated helmet on a safety poster (no person) is attenuated and rejected."""
        # Poster helmet: high confidence 0.85, but no person near it
        poster_hat = [[600, 100, 680, 180, 0.85, 0]]  # x1, y1, x2, y2, conf, cls
        persons = []  # no persons

        filtered_hats, assoc = filter_poster_false_positives(
            poster_hat,
            persons,
            overlap_thresh=0.25,
            kappa=0.15,
            conf_thresh=0.25,
        )

        assert len(filtered_hats) == 0, "Poster false positive should be rejected!"
        assert len(assoc) == 1
        assert assoc[0]["is_poster_rejected"] is True
        # 0.85 * 0.15 = 0.1275 < 0.25 threshold
        assert assoc[0]["effective_score"] == pytest.approx(0.85 * 0.15, rel=1e-3)

    def test_filter_real_worker_helmet_retention(self):
        """Real worker wearing a helmet passes filter with unmodified score."""
        real_hat = [[150, 100, 250, 180, 0.92, 0]]
        real_person = [[100, 100, 300, 500, 0.88, 1]]

        filtered_hats, assoc = filter_poster_false_positives(
            real_hat,
            real_person,
            overlap_thresh=0.25,
            kappa=0.15,
            conf_thresh=0.25,
        )

        assert len(filtered_hats) == 1, "Real helmet must be retained!"
        assert filtered_hats[0, 4] == pytest.approx(0.92, rel=1e-3)
        assert assoc[0]["is_supported"] is True
        assert assoc[0]["max_overlap"] >= 0.25

    def test_mixed_scene_filtering(self):
        """Scene with 2 real workers with helmets and 1 poster helmet."""
        hats = [
            [120, 100, 200, 170, 0.95, 0],   # Hat 1 on Worker 1 -> Retain
            [420, 120, 500, 190, 0.89, 0],   # Hat 2 on Worker 2 -> Retain
            [800, 50, 880, 120, 0.80, 0],    # Hat 3 on Poster -> Reject!
        ]
        persons = [
            [100, 100, 220, 450, 0.90, 1],   # Worker 1
            [400, 110, 520, 480, 0.85, 1],   # Worker 2
        ]

        filtered_hats, assoc = filter_poster_false_positives(
            hats,
            persons,
            overlap_thresh=0.25,
            kappa=0.15,
            conf_thresh=0.25,
        )

        assert len(filtered_hats) == 2, f"Expected 2 confirmed helmets, got {len(filtered_hats)}"
        assert assoc[0]["is_supported"] is True
        assert assoc[1]["is_supported"] is True
        assert assoc[2]["is_supported"] is False
        assert assoc[2]["is_poster_rejected"] is True

    def test_edge_case_empty_inputs(self):
        """Filter handles completely empty inputs without crashing."""
        empty_hats = []
        empty_persons = []
        filtered, assoc = filter_poster_false_positives(empty_hats, empty_persons)
        assert len(filtered) == 0
        assert len(assoc) == 0

    def test_edge_case_torch_tensors(self):
        """Filter works seamlessly when given PyTorch CUDA/CPU tensors."""
        hat_tensor = torch.tensor([[150.0, 100.0, 250.0, 180.0, 0.90, 0.0]])
        person_tensor = torch.tensor([[100.0, 100.0, 300.0, 500.0, 0.85, 1.0]])

        filtered, assoc = filter_poster_false_positives(hat_tensor, person_tensor)
        assert len(filtered) == 1
        assert isinstance(filtered, np.ndarray)

    def test_boundary_overlap_threshold(self):
        """Helmet with overlap exactly at threshold (0.25) is supported."""
        # Hat: width 100, height 100 (area 10000)
        # Person upper torso overlaps exactly 25 width x 100 height = 2500 (ratio 0.25)
        p_box = [175, 100, 300, 500]
        h_box = [100, 100, 200, 200, 0.70]

        filtered, assoc = filter_poster_false_positives(
            [h_box],
            [p_box],
            overlap_thresh=0.25,
            conf_thresh=0.25,
        )
        assert len(filtered) == 1
        assert assoc[0]["is_supported"] is True

    def test_lower_body_helmet_rejected(self):
        """A helmet graphic drawn near feet or knees is rejected under upper_torso constraint."""
        # Person: [100, 100, 300, 500] (upper torso = 100 to 340)
        # Hat at feet: [150, 420, 250, 480] (y in 420-480, far from head)
        p_box = [100, 100, 300, 500]
        feet_hat = [150, 420, 250, 480, 0.75]

        filtered, assoc = filter_poster_false_positives(
            [feet_hat],
            [p_box],
            overlap_thresh=0.25,
            use_upper_torso=True,
            conf_thresh=0.25,
        )
        assert len(filtered) == 0, "Helmet near feet must be rejected as unnatural position"
        assert assoc[0]["is_supported"] is False

    def test_degenerate_inverted_boxes(self):
        """Test that inverted bounding boxes (x2 < x1 or y2 < y1) return 0 ratio and don't crash."""
        p_box = [100, 100, 300, 500]
        inverted_hat = [250, 200, 150, 100]  # inverted x and y
        ratio = compute_box_intersection_ratio(inverted_hat, p_box)
        assert ratio == 0.0

    def test_zero_area_box(self):
        """Test line or point boxes with zero area return 0 ratio."""
        p_box = [100, 100, 300, 500]
        line_hat = [150, 100, 150, 200]  # width 0
        ratio = compute_box_intersection_ratio(line_hat, p_box)
        assert ratio == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
