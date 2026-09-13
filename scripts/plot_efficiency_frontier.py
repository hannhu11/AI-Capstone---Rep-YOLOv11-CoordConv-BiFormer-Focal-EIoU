import matplotlib.pyplot as plt

# Set IEEE font style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

data = [
    {'model': 'YOLOv8n', 'latency': 2.85, 'map50_95': 60.26, 'type': 'Baseline'},
    {'model': 'YOLOv8s', 'latency': 6.10, 'map50_95': 62.21, 'type': 'Baseline'},
    {'model': 'YOLOv10n', 'latency': 2.66, 'map50_95': 60.35, 'type': 'Baseline'},
    {'model': 'YOLOv10s', 'latency': 6.23, 'map50_95': 62.19, 'type': 'Baseline'},
    {'model': 'YOLO11n', 'latency': 2.58, 'map50_95': 60.82, 'type': 'Baseline'},
    {'model': 'YOLO11s', 'latency': 5.82, 'map50_95': 62.38, 'type': 'Baseline'},
    {'model': 'Rep-YOLO11s (Ours)', 'latency': 2.92, 'map50_95': 62.54, 'type': 'Proposed'}
]

fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

# Plot standard baselines on SHWD
baselines = [d for d in data if d['type'] == 'Baseline']
for b in baselines:
    ax.scatter(b['latency'], b['map50_95'], color='#1f77b4', s=70, zorder=3)

# Offset annotations cleanly to avoid overlapping
offsets = {
    'YOLOv8n': (0.15, -0.35),
    'YOLOv8s': (-0.95, -0.45),
    'YOLOv10n': (-1.05, -0.25),
    'YOLOv10s': (0.15, -0.45),
    'YOLO11n': (-0.95, 0.22),
    'YOLO11s': (0.15, 0.22),
}

for b in baselines:
    dx, dy = offsets.get(b['model'], (0.12, 0.12))
    ax.annotate(b['model'], (b['latency'], b['map50_95']),
                xytext=(b['latency'] + dx, b['map50_95'] + dy),
                fontsize=9, fontweight='normal',
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5) if abs(dx)>0.5 else None)

# Plot Proposed Rep-YOLO11s (Ours)
our_model = [d for d in data if d['type'] == 'Proposed'][0]
ax.scatter(our_model['latency'], our_model['map50_95'], color='#d62728', marker='*', s=220, zorder=5, label='Rep-YOLO11s (Ours)')
ax.annotate("Rep-YOLO11s (Ours)\n2.92 ms / 62.54%", 
            (our_model['latency'], our_model['map50_95']),
            xytext=(our_model['latency'] + 0.18, our_model['map50_95'] + 0.30),
            fontsize=9.5, fontweight='bold', color='#d62728')

# Reference lines
ax.axvline(x=5.0, color='gray', linestyle='--', linewidth=0.8, label='Latency Threshold (5.0 ms)')
ax.set_xlabel('Pure GPU Inference Latency (ms) [Lower is Better]', fontsize=10.5, fontweight='bold')
ax.set_ylabel('mAP@0.5:0.95 (%) [Higher is Better]', fontsize=10.5, fontweight='bold')
ax.set_title('Efficiency Frontier: Accuracy vs Inference Latency', fontsize=11, fontweight='bold', pad=10)

ax.set_xlim(1.0, 7.5)
ax.set_ylim(59.5, 63.8)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('paper_overleaf/figures/shwd_latency_vs_map.png', dpi=300)
plt.savefig('paper_overleaf/figures/shwd_latency_vs_map.pdf')
print('Generated crisp IEEE vector/PNG Efficiency Frontier plot successfully!')
