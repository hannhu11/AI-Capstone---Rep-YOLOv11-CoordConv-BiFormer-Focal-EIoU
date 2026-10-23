with open('scripts/generate_experiment_notebooks.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for idx in range(len(lines)):
    if 'Flycam & Drone' in lines[idx] and idx > 700:
        lines[idx] = '    print("[PROCESSING] Đang xử lý IITU Safety-Helmet (Flycam & Drone)...")\n'
    if '[VERIFIED] Tổng số mẫu hợp nhất' in lines[idx] and idx > 700:
        lines[idx] = 'print(f"[VERIFIED] Tổng số mẫu hợp nhất: {len(unified_samples)} ảnh")\n'
    if '[OBJECTIVE] KẾT QUẢ TỔNG QUAN' in lines[idx] and idx > 700:
        lines[idx] = 'print(f"[OBJECTIVE] KẾT QUẢ TỔNG QUAN:")\n'
    if '[REPORT] KẾT QUẢ CHI TIẾT THEO TỪNG LỚP TRONG BỘ 4' in lines[idx] and idx > 700:
        lines[idx] = 'print(f"[REPORT] KẾT QUẢ CHI TIẾT THEO TỪNG LỚP TRONG BỘ 4:")\n'

with open('scripts/generate_experiment_notebooks.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed lines successfully!")
