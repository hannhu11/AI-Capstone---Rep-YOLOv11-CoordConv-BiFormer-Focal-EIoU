with open('scripts/generate_experiment_notebooks.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace the inner triple double quotes with escaped triple double quotes in Code 6 of Bo4
target1 = 'report_md = f"""# [REPORT] BÁO CÁO KẾT QUẢ THỰC NGHIỆM TASK 2 - BỘ 4 (UNIFIED 7-CLASS MASTER)'
replacement1 = 'report_md = f\\\"\\\"\\\"# [REPORT] BÁO CÁO KẾT QUẢ THỰC NGHIỆM TASK 2 - BỘ 4 (UNIFIED 7-CLASS MASTER)'

target2 = '2. **Cơ chế Annotation Fusion vượt trội**: Đã xử lý triệt để 770 ảnh trùng lặp giữa GDUT và CHV, loại bỏ hoàn toàn nguy cơ rò rỉ dữ liệu (0% Data Leakage).\\n\"\"\"\\n\\nwith open(\"/kaggle/working/BAO_CAO_BO4_UNIFIED_PPE_7CLASS.md\"'
# Or just search and replace the closing quote before 'with open("/kaggle/working/BAO_CAO_BO4_UNIFIED_PPE_7CLASS.md"'

if target1 in text:
    text = text.replace(target1, replacement1)
    print("Replaced target1!")
else:
    print("target1 NOT FOUND!")

# Find closing quote
pos = text.find('with open("/kaggle/working/BAO_CAO_BO4_UNIFIED_PPE_7CLASS.md"')
if pos != -1:
    # look backwards for """
    q_pos = text.rfind('"""', 0, pos)
    if q_pos != -1:
        text = text[:q_pos] + '\\"\\"\\"' + text[q_pos+3:]
        print("Replaced closing quote successfully!")

with open('scripts/generate_experiment_notebooks.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Done fixing script quotes.")
