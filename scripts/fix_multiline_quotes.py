with open('scripts/generate_experiment_notebooks.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix yaml_content inside Code 3 of Bo4
old_yaml = '''yaml_content = f"""# UNIFIED PPE 7-CLASS MASTER CONFIGURATION
path: {TARGET_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 7
names: ['blue_helmet', 'green_helmet', 'red_helmet', 'white_helmet', 'yellow_helmet', 'no_helmet', 'safety_vest']
"""'''

new_yaml = '''yaml_content = f\'\'\'# UNIFIED PPE 7-CLASS MASTER CONFIGURATION
path: {TARGET_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 7
names: ['blue_helmet', 'green_helmet', 'red_helmet', 'white_helmet', 'yellow_helmet', 'no_helmet', 'safety_vest']
\'\'\''''

# Fix report_md inside Code 6 of Bo4
# Find from report_md = to with open
start_rep = text.find('report_md = f')
end_rep = text.find('with open("/kaggle/working/BAO_CAO_BO4_UNIFIED_PPE_7CLASS.md"')
if start_rep != -1 and end_rep != -1:
    rep_block = text[start_rep:end_rep]
    # Replace the quote delimiters with triple single quotes
    # Let's inspect what rep_block looks like
    print("Found report block length:", len(rep_block))

text = text.replace(old_yaml, new_yaml)

# Let's do report_md cleanly:
text = text.replace('report_md = f\\\"\\\"\\\"# [REPORT]', "report_md = f\'\'\'# [REPORT]")
text = text.replace('report_md = f"""# [REPORT]', "report_md = f\'\'\'# [REPORT]")
text = text.replace('(0% Data Leakage).\n\\\"\\\"\\\"', "(0% Data Leakage).\n\'\'\'")
text = text.replace('(0% Data Leakage).\n"""', "(0% Data Leakage).\n\'\'\'")

with open('scripts/generate_experiment_notebooks.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Applied single quote fixes!")
