import json
import re
from pathlib import Path

notebooks = [
    Path("ppe_extension_experiment/notebooks/Task2_Account_1_DataPipeline_and_Baselines.ipynb"),
    Path("ppe_extension_experiment/notebooks/Task2_Account_2_Modular_Ablations_A1_to_A5.ipynb"),
    Path("ppe_extension_experiment/notebooks/Task2_Account_3_Proposed_Champion_A6_and_Failure_Analysis.ipynb")
]

emoji_pattern = re.compile(
    "[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]|[\u200d\ufe0f]"
)

all_clean = True
for nb_path in notebooks:
    assert nb_path.exists(), f"Missing {nb_path}"
    content = nb_path.read_text(encoding='utf-8')
    data = json.loads(content)
    emojis = emoji_pattern.findall(content)
    print(f"File: {nb_path.name}")
    print(f"  - Total cells: {len(data['cells'])}")
    print(f"  - Emojis found: {len(emojis)}")
    if emojis:
        all_clean = False
        print(f"  - Discovered emojis: {set(emojis)}")
    else:
        print("  - [PASS] Zero emojis / icons confirmed!")

if all_clean:
    print("\n[ALL PASS] All 3 notebooks are valid JSON and 100% free of emojis!")
else:
    print("\n[FAIL] Found emojis that need cleanup.")
