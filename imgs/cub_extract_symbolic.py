from pathlib import Path

# Path to the unzipped official dataset folder containing images/, images.txt, train_test_split.txt, etc.
CUB_ROOT = Path("/archive/datasets/CUB_200_2011").resolve()

# Output root for the symlinked train/ and test/ trees
OUT_ROOT = Path("./CUB_200_2011/")

# Create output root (no-op if exists)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

print("CUB_ROOT:", CUB_ROOT)
print("OUT_ROOT:", OUT_ROOT)

def read_two_column_txt(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                left = int(parts[0])
                right = parts[1]
                # right may be int-like for split, or a filepath string for images
                rows.append((left, right))
    return rows

images_txt = CUB_ROOT / "images.txt"                 # <img_id> <relpath>
split_txt  = CUB_ROOT / "train_test_split.txt"       # <img_id> <is_train:0/1>

images_map = dict(read_two_column_txt(images_txt))           # id -> '001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
split_map  = {i: int(s) for i, s in read_two_column_txt(split_txt)}  # id -> 1 if train else 0

print("images:", len(images_map), "split:", len(split_map))

import os

SRC_IMAGES = CUB_ROOT / "images"

def make_relative_link(src_abs: Path, link_dir: Path, link_name: str):
    """Create a relative symlink link_dir/link_name -> src_abs."""
    link_dir.mkdir(parents=True, exist_ok=True)
    link_path = link_dir / link_name
    if link_path.exists() or link_path.is_symlink():
        return link_path
    # compute relative target from link_dir to src_abs
    rel_target = os.path.relpath(src_abs, start=link_dir)
    try:
        link_path.symlink_to(rel_target)  # file symlink
    except OSError as e:
        # Optional fallback: try making a hardlink if symlink not allowed
        try:
            os.link(src_abs, link_path)
        except OSError:
            print(f"Failed to link: {link_path} -> {rel_target} ({e})")
    return link_path

created = 0
missing = 0

for img_id, relpath in images_map.items():
    split_name = "train" if split_map[img_id] == 1 else "test"
    rel = Path(relpath)          # e.g., 'Black_footed_Albatross/...jpg'
    class_dir = rel.parts[0]         # e.g., 'Black_footed_Albatross'
    src_abs = SRC_IMAGES / rel       # absolute source path in dataset
    if not src_abs.exists():
        missing += 1
        continue
    dst_dir = OUT_ROOT / split_name / class_dir
    link_path = make_relative_link(src_abs, dst_dir, rel.name)
    created += 1

print(f"Links created: {created}, missing sources: {missing}")
print("Train root:", OUT_ROOT / "train")
print("Test  root:", OUT_ROOT / "test")

import itertools

def sample_links(root: Path, n=5):
    files = [p for p in root.rglob("*") if p.is_file()]
    for p in files[:n]:
        try:
            print(p, "->", p.resolve())
        except Exception as e:
            print("Failed to resolve:", p, e)

print("Train examples:")
sample_links(OUT_ROOT / "train", n=5)
print("\nTest examples:")
sample_links(OUT_ROOT / "test", n=5)
