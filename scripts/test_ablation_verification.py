import sys
sys.path.append('.')
import torch
from scripts.ablation_factory import build_ablation_model

print("Testing build_ablation_model on all ablations...")

# Test A0
m0 = build_ablation_model("A0", "yolo11s.pt")
print("A0 created successfully, num layers:", len(m0.model.model))

# Test A1
m1 = build_ablation_model("A1", "yolo11s.pt", nc=6)
print("A1 created successfully, num layers:", len(m1.model.model))

# Test A2
m2 = build_ablation_model("A2", "yolo11s.pt", nc=6)
print("A2 created successfully, num layers:", len(m2.model.model))

# Test A3
m3 = build_ablation_model("A3", "yolo11s.pt", nc=6)
print("A3 created successfully, num layers:", len(m3.model.model))

# Test A5
m5 = build_ablation_model("A5", "yolo11s.pt", nc=6)
print("A5 created successfully, num layers:", len(m5.model.model))

# Test A6
m6 = build_ablation_model("A6", "yolo11s.pt", nc=6)
print("A6 created successfully, num layers:", len(m6.model.model))

# Test forward pass with dummy tensor
x = torch.randn(1, 3, 640, 640)
print("Running forward passes on dummy tensor (1, 3, 640, 640)...")

m0.model.eval()
with torch.no_grad():
    out0 = m0.model(x)
print("A0 forward pass ok, output type:", type(out0))

# Test stem L2 diff between m0 and m2
m2.model.eval()
with torch.no_grad():
    stem0 = m0.model.model[0](x)
    stem2 = m2.model.model[0](x)
    stem_diff = (stem2 - stem0).norm() / stem0.norm()
    print(f"Stem (CoordConv vs Conv0) relative L2 diff: {stem_diff.item() * 100:.4f}%")

# Test RepConv layer diff between m0 and m3 at layer 1
m3.model.eval()
with torch.no_grad():
    # layer 1 is 3x3 Conv
    l1_m0 = m0.model.model[1](stem0)
    l1_m3 = m3.model.model[1](stem0)
    l1_diff = (l1_m3 - l1_m0).norm() / l1_m0.norm()
    print(f"Layer 1 (RepConv vs Conv) relative L2 diff: {l1_diff.item() * 100:.4f}%")

# Test BiFormer wrapper diff between m0 and m5 at layer 10 (or first replaced layer)
m5.model.eval()
# find replaced layer index in m5
biformer_idx = None
for i, l in enumerate(m5.model.model):
    if l.__class__.__name__ == "BiFormerSequentialWrapper":
        biformer_idx = i
        break
print(f"m5 BiFormer wrapper is at index {biformer_idx}")
if biformer_idx is not None:
    wrapper = m5.model.model[biformer_idx]
    # Test wrapper input vs output of orig_layer
    dummy_feat = torch.randn(1, wrapper.biformer.channels, 20, 20)
    with torch.no_grad():
        orig_out = wrapper.orig_layer(dummy_feat) if hasattr(wrapper.orig_layer, '__call__') else dummy_feat
        wrap_out = wrapper.biformer(orig_out)
        b_diff = (wrap_out - orig_out).norm() / orig_out.norm()
        print(f"BiFormer wrapper step-0 feature diff: {b_diff.item() * 100:.4f}%")
