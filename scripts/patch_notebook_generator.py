with open('scripts/generate_experiment_notebooks.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix in bo1
old_bo1_cell6 = '''# CELL 6: TẠO FILE BÁO CÁO KẾT QUẢ ĐẠT CHUẨN TASK 2
report_md = f"""# [REPORT] BÁO CÁO KẾT QUẢ THỰC NGHIỆM TASK 2 - BỘ 1 (GDUT-HWD 5-CLASS)'''
new_bo1_cell6 = '''# CELL 6: TẠO FILE BÁO CÁO KẾT QUẢ ĐẠT CHUẨN TASK 2
def get_p(idx):
    return float(metrics.box.p[idx]*100) if hasattr(metrics.box, 'p') and idx < len(metrics.box.p) else 0.0
def get_r(idx):
    return float(metrics.box.r[idx]*100) if hasattr(metrics.box, 'r') and idx < len(metrics.box.r) else 0.0
def get_ap50(idx):
    return float(metrics.box.ap50[idx]*100) if hasattr(metrics.box, 'ap50') and idx < len(metrics.box.ap50) else 0.0

report_md = f"""# [REPORT] BÁO CÁO KẾT QUẢ THỰC NGHIỆM TASK 2 - BỘ 1 (GDUT-HWD 5-CLASS)'''
text = text.replace(old_bo1_cell6, new_bo1_cell6, 1)

# Fix in bo3
old_bo3_cell6 = '''# CELL 6: TẠO FILE BÁO CÁO KẾT QUẢ FLYCAM ĐẠT CHUẨN TASK 2
report_md = f"""# [REPORT] BÁO CÁO KẾT QUẢ THỰC NGHIỆM TASK 2 - BỘ 3 (IITU DRONE 5-CLASS)'''
new_bo3_cell6 = '''# CELL 6: TẠO FILE BÁO CÁO KẾT QUẢ FLYCAM ĐẠT CHUẨN TASK 2
def get_p(idx):
    return float(metrics.box.p[idx]*100) if hasattr(metrics.box, 'p') and idx < len(metrics.box.p) else 0.0
def get_r(idx):
    return float(metrics.box.r[idx]*100) if hasattr(metrics.box, 'r') and idx < len(metrics.box.r) else 0.0
def get_ap50(idx):
    return float(metrics.box.ap50[idx]*100) if hasattr(metrics.box, 'ap50') and idx < len(metrics.box.ap50) else 0.0

report_md = f"""# [REPORT] BÁO CÁO KẾT QUẢ THỰC NGHIỆM TASK 2 - BỘ 3 (IITU DRONE 5-CLASS)'''
text = text.replace(old_bo3_cell6, new_bo3_cell6, 1)

# Fix in bo4
old_bo4_cell6 = '''# CELL 6: TẠO FILE BÁO CÁO KẾT QUẢ HỢP NHẤT ĐẠT CHUẨN TASK 2
report_md = f"""# [REPORT] BÁO CÁO KẾT QUẢ THỰC NGHIỆM TASK 2 - BỘ 4 (UNIFIED 7-CLASS MASTER)'''
new_bo4_cell6 = '''# CELL 6: TẠO FILE BÁO CÁO KẾT QUẢ HỢP NHẤT ĐẠT CHUẨN TASK 2
def get_p(idx):
    return float(metrics.box.p[idx]*100) if hasattr(metrics.box, 'p') and idx < len(metrics.box.p) else 0.0
def get_r(idx):
    return float(metrics.box.r[idx]*100) if hasattr(metrics.box, 'r') and idx < len(metrics.box.r) else 0.0
def get_ap50(idx):
    return float(metrics.box.ap50[idx]*100) if hasattr(metrics.box, 'ap50') and idx < len(metrics.box.ap50) else 0.0

report_md = f"""# [REPORT] BÁO CÁO KẾT QUẢ THỰC NGHIỆM TASK 2 - BỘ 4 (UNIFIED 7-CLASS MASTER)'''
text = text.replace(old_bo4_cell6, new_bo4_cell6, 1)

# Replace table lines in bo4
for i, name in enumerate(['blue_helmet` (Xanh dương)', 'green_helmet` (Xanh lá)', 'red_helmet` (Đỏ)', 'white_helmet` (Trắng)', 'yellow_helmet` (Vàng)', 'no_helmet` (Vi phạm)', 'safety_vest` (Áo phản quang)']):
    target = f'| {i} | `{name} | {{metrics.box.p[{i}]*100:.2f if {i} < len(metrics.box.p) else 0.0}} | {{metrics.box.r[{i}]*100:.2f if {i} < len(metrics.box.r) else 0.0}} | {{metrics.box.ap50[{i}]*100:.2f if {i} < len(metrics.box.ap50) else 0.0}} |'
    replacement = f'| {i} | `{name} | {{get_p({i}):.2f}}% | {{get_r({i}):.2f}}% | {{get_ap50({i}):.2f}}% |'
    text = text.replace(target, replacement)

with open('scripts/generate_experiment_notebooks.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated scripts/generate_experiment_notebooks.py successfully!')
