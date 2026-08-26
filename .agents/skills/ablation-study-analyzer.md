# Skill Name: ablation-study-analyzer
# Description: Tự động tổng hợp kết quả Ablation Study và diễn giải vai trò của từng thành phần kiến trúc (BiFormer, RepConv, CoordConv, Focal Loss).

## Core Rules:
1. Khi phân tích bảng Ablation Study:
   - Diễn giải mức độ đóng góp của từng module ($A_0 \rightarrow A_1 \rightarrow ... \rightarrow A_6$).
   - Phân tích sự đánh đổi (Trade-off) giữa $mAP_{50}$, $mAP_{50-95}$, Latency (ms), và FPS.
   - Giải thích lý do về mặt lý thuyết toán học/kiến trúc tại sao module đó giúp tăng hiệu năng (e.g., "RepConv merges multi-branch paths into a single 3x3 conv during deployment, preserving expressive power while reducing memory access cost").