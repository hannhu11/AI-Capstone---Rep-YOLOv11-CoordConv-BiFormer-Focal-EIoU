import re

with open('scripts/generate_experiment_notebooks.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Fix print with \n in Cell 2 of bo4
text = text.replace(
    'print(\\"[PROCESSING] Đang xử lý IITU Safety-Helmet (Flycam & Drone)...\\\\n\\")',
    'print(\\"[PROCESSING] Đang xử lý IITU Safety-Helmet (Flycam & Drone)...\\")'
)
text = text.replace(
    'print(f\\"\\\\n[VERIFIED] Tổng số mẫu hợp nhất: {len(unified_samples)} ảnh\\")',
    'print(f\\"[VERIFIED] Tổng số mẫu hợp nhất: {len(unified_samples)} ảnh\\")'
)

# 2. Fix f.write('\n'.join(bboxes) + '\n') in Cell 3 of bo4
text = text.replace(
    "f.write('\\n'.join(bboxes) + '\\n')",
    "f.write('\\\\n'.join(bboxes) + '\\\\n')"
)

# 3. Fix print with \n in Cell 5 of bo4
text = text.replace(
    'print(f\\"\\\\n[OBJECTIVE] KẾT QUẢ TỔNG QUAN:\\")',
    'print(f\\"[OBJECTIVE] KẾT QUẢ TỔNG QUAN:\\")'
)
text = text.replace(
    'print(f\\"\\\\n[REPORT] KẾT QUẢ CHI TIẾT THEO TỪNG LỚP TRONG BỘ 4:\\")',
    'print(f\\"[REPORT] KẾT QUẢ CHI TIẾT THEO TỪNG LỚP TRONG BỘ 4:\\")'
)

with open('scripts/generate_experiment_notebooks.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Applied string fixes in scripts/generate_experiment_notebooks.py")
