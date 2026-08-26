import datetime
from pathlib import Path

def find_all_engines():
    base_dir = Path("c:/Users/ADMIN/Downloads/capstone AI")
    print(f"Scanning for all .engine files in: {base_dir}\n")
    
    engine_files = list(base_dir.rglob("*.engine"))
    
    print(f"Found {len(engine_files)} .engine file(s):\n")
    for idx, f in enumerate(engine_files, 1):
        mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"{idx}. Path     : {f}")
        print(f"   Size     : {size_mb:.2f} MB")
        print(f"   Modified : {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)

if __name__ == "__main__":
    find_all_engines()
