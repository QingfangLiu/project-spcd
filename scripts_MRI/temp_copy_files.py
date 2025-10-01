
"""
Script to selectively duplicate the SPCD_data folder.

- Copies all subfolders and files under SPCD_data except for the "nii" folder.
- For the "nii" folder:
    * Preserves the original folder structure.
    * Copies only files with a .tsv extension.
- Ensures that copied files and directories keep their relative paths.
- Uses shutil.copytree for full directories and shutil.copy2 for individual files.

Result: A new folder (SPCD_data_copy) containing a full duplicate of SPCD_data,
        with only .tsv files retained under the nii/ directory.
"""


import shutil
import os
from pathlib import Path

# ====== INPUT / OUTPUT PATHS ======
src = Path("/Users/liuq13/SPCD_data")       # your original SPCD_data folder
dst = Path("/Volumes/X9Pro/SPCD_data_copy")  # destination folder

dst.mkdir(parents=True, exist_ok=True)

for item in src.iterdir():
    if item.is_dir():
        if item.name == "nii":
            # Walk through nii subfolders
            for root, _, files in os.walk(item):
                root_path = Path(root)
                rel_path = root_path.relative_to(src)  # relative path under SPCD_data
                target_dir = dst / rel_path
                target_dir.mkdir(parents=True, exist_ok=True)
                for f in files:
                    if f.endswith(".tsv"):
                        shutil.copy2(root_path / f, target_dir / f)
        else:
            # Copy entire folder recursively
            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
    elif item.is_file():
        shutil.copy2(item, dst / item.name)

print("✅ Copy completed with nii/.tsv files and folder structure preserved.")
