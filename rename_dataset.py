import os
import re
import shutil
from pathlib import Path
from tqdm import tqdm
from config import DATASET_DIR

def rename_augments():
    #\"\"\"Rename dataset/*/*_XXXX_aug_i.jpg → name_(XXXX*10 + i):05d.jpg\"\"\"
    renames = []
    
    for person_dir in DATASET_DIR.iterdir():
        if not person_dir.is_dir():
            continue
            
        print(f"Processing {person_dir.name}/")
        
        for img_path in person_dir.glob("*.jpg"):
            filename = img_path.name
            match = re.match(r'(.+)_(\d{4})_aug_(\d)\.jpg$', filename)
            if not match:
                continue  # Skip originals
                
            person, base_str, aug_str = match.groups()
            base_num = int(base_str)
            aug_i = int(aug_str)
            new_num = base_num * 10 + aug_i
            new_name = f"{person}_{new_num:05d}.jpg"
            new_path = img_path.parent / new_name
            
            if new_path.exists():
                print(f"Warning: {new_name} already exists, skipping {filename}")
                continue
                
            renames.append((img_path, new_path))
    
    print(f"\nFound {len(renames)} files to rename. Proceeding...")
    
    for old, new in tqdm(renames, desc="Renaming"):
        os.rename(old, new)
        print(f"Renamed: {old.name} → {new.name}")
    
    print("✅ Dataset rename completed!")

if __name__ == "__main__":
    rename_augments()

