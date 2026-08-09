import os
import re

# Change this to your actual folder path
folder = r"C:\Users\gbhan\COLLEGE NOTES ARRANGING\SECOND SEMESTER\Image-Processing and Computer Vision\TASKS\Task 4"

# Supported image extensions
valid_ext = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

# Get all image files sorted by modification time (preserves capture order)
files = [
    f for f in os.listdir(folder)
    if os.path.splitext(f)[1] in valid_ext
]
files.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)))

print(f"Found {len(files)} images")

# Rename
renamed = []
for i, old_name in enumerate(files, start=1):
    ext = os.path.splitext(old_name)[1].lower()  # normalize to lowercase .jpg
    new_name = f"image_{i:03d}{ext}"
    old_path = os.path.join(folder, old_name)
    new_path = os.path.join(folder, new_name)

    if old_name == new_name:
        print(f"  [SKIP] {old_name} already correctly named")
        continue

    os.rename(old_path, new_path)
    renamed.append((old_name, new_name))
    print(f"  {old_name}  →  {new_name}")

print(f"\nDone! Renamed {len(renamed)} files.")

# Also generate a starter labels CSV
import csv
csv_path = os.path.join(folder, "labels.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "TL", "TR", "BL", "BR"])
    for _, new_name in renamed:
        writer.writerow([new_name, "", "", "", ""])
    # Also add any files that were already correctly named
    for f_name in files:
        ext = os.path.splitext(f_name)[1].lower()
        expected = f"image_{files.index(f_name)+1:03d}{ext}"
        if f_name == expected:
            pass  # already handled above won't be in renamed

# Rewrite CSV cleanly with all files
all_images = sorted([
    f for f in os.listdir(folder)
    if os.path.splitext(f)[1] in {".jpg", ".jpeg", ".png"}
])
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "TL", "TR", "BL", "BR"])
    for img in all_images:
        writer.writerow([img, "", "", "", ""])

print(f"\nCreated blank labels.csv at:\n{csv_path}")
print(f"Total images in CSV: {len(all_images)}")
