import os
import re
from pathlib import Path
from tqdm import tqdm

def batch_rename_hieu():
    source_dir = Path('predata/hieu00')
    target_dir = Path('predata/ThanHieu1')
    
    if not source_dir.exists():
        print("❌ Source dir not found:", source_dir)
        return
    
    target_dir.mkdir(exist_ok=True)
    
    # Find highest number in target dir
    max_num = -1
    for img_path in target_dir.glob('*.jpg'):
        filename = img_path.name
        # Match ThanHieu1_XXXX.jpg or thanhieu1_XXXX.jpg
        match = re.match(r'(?:ThanHieu1|thanhieu1)_(\d{4})\.jpg$', filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    start_num = max_num + 1
    print(f"Target next number: {start_num}")
    
    # List source files, sort numerically
    source_files = sorted(source_dir.glob('hieu00_*.jpg'), 
                         key=lambda p: int(re.match(r'hieu00_(\d{4})\.jpg$', p.name).group(1)))
    
    if not source_files:
        print("❌ No source files found.")
        return
    
    print(f"Found {len(source_files)} files to rename (starting from {start_num:04d}).")
    
    renames = []
    for old_path in source_files:
        i = len(renames)
        new_num = start_num + i
        new_name = f'ThanHieu1_{new_num:04d}.jpg'
        new_path = target_dir / new_name
        if new_path.exists():
            print(f"⚠️ Skipping {old_path.name} → {new_name} (exists)")
            continue
        renames.append((old_path, new_path))
    
    print(f"\nPreview: {len(renames)} renames ready. Proceeding...")
    print("Sample:", renames[0][0].name, "→", renames[0][1].name)
    
    for old, new in tqdm(renames, desc="Renaming"):
        os.rename(old, new)
        print(f"Renamed: {old.name} → {new.name}")
    
    print(f"✅ Completed! Renamed {len(renames)}/{len(source_files)} files.")

if __name__ == "__main__":
    batch_rename_hieu()
