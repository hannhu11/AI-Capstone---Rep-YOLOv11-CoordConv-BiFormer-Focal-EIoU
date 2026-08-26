import shutil
import zipfile
from pathlib import Path

def prepare_v2_zip():
    base_dir = Path("c:/Users/ADMIN/Downloads/capstone AI")
    target_dir = base_dir / "kaggle_upload_shwd_benchmark_code_v2"
    target_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        ("kaggle_stage3_experiments.py", "kaggle_stage3_experiments.py"),
        ("kaggle_shwd_baseline.py", "kaggle_shwd_baseline.py"),
        ("custom_ablation_modules.py", "custom_ablation_modules.py"),
        ("albumentations_hardcase_policy.py", "albumentations_hardcase_policy.py"),
        ("scripts/generate_gradcam_comparison.py", "generate_gradcam_comparison.py"),
    ]

    print("-> Packing files into kaggle_upload_shwd_benchmark_code_v2:")
    for src_rel, dst_name in files_to_copy:
        src_path = base_dir / src_rel
        dst_path = target_dir / dst_name
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"   [+] Copied: {src_rel} -> {dst_name}")
        else:
            print(f"   [!] Missing: {src_path}")

    # Also copy scripts folder inside for compatibility
    scripts_dst = target_dir / "scripts"
    scripts_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_dir / "scripts/generate_gradcam_comparison.py", scripts_dst / "generate_gradcam_comparison.py")

    # Zip target folder
    zip_path = base_dir / "shwd-benchmark-code-v2.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_out:
        for f in target_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(target_dir)
                zip_out.write(f, arcname=str(rel))

    print("\n" + "=" * 65)
    print("✅ SHWD-BENCHMARK-CODE VERSION 2 ZIP CREATED SUCCESSFULLY!")
    print(f"   - Zip Path : {zip_path}")
    print(f"   - Size     : {zip_path.stat().st_size / 1024:.2f} KB")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    prepare_v2_zip()
