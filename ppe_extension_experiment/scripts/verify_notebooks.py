import json
import re
from pathlib import Path

notebooks = [
    Path("ppe_extension_experiment/notebooks/DataPipeline_and_Baselines_chv.ipynb"),
    Path("ppe_extension_experiment/notebooks/Modular_Ablations_A1_to_A5_chv.ipynb"),
    Path("ppe_extension_experiment/notebooks/Proposed_Champion_A6_and_Failure_Analysis_chv.ipynb")
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
    print("\n[ALL PASS] All 3 renamed notebooks are valid JSON and 100% free of emojis!")
else:
    print("\n[FAIL] Found emojis that need cleanup.")
