"""
rate_images.py

Interactive image annotation tool for collecting human blur ratings.

This script loads all `.jpg` images from subfolders in the current directory,
displays them one at a time, and collects subjective blur ratings from a user
on a 3-point ordinal scale:

    0 = sharp
    1 = slight blur
    2 = blurred

The user is prompted for their rater name (e.g., 'rater1'), and ratings are stored
in an Excel file that is updated incrementally. Already-rated images are skipped.

Inputs:
- JPEG images organized in subfolders
- Optional: Existing 'image_rating.xlsx' file with previous ratings

Outputs:
- image_rating.xlsx : Excel file with image names, folder names, and per-rater scores

Requirements:
- pandas, opencv-python (cv2)

Note:
- Images are resized to fit within an 800x800 display window.
- Works best when run in a local Python environment with GUI support.
"""

import os
import pandas as pd
import cv2

# ========== Configuration ==========
image_dir = os.getcwd()
excel_file = "image_rating.xlsx"
participant_name = input("Enter your name or ID (e.g., rater1): ").strip()

# Find all subfolders (assumed to contain images)
subfolders = [f.name for f in os.scandir(image_dir) if f.is_dir()]

# ========== Load or create dataset ==========
if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    image_list = []
    for subfolder in subfolders:
        for fname in os.listdir(subfolder):
            if fname.lower().endswith(".jpg"):
                image_list.append({
                    "Image Name": fname,
                    "Folder": subfolder
                })
    df = pd.DataFrame(image_list)

# Add column if not already present
if participant_name in df.columns:
    print(f"Participant '{participant_name}' already exists. Previous values will be overwritten.")
else:
    df[participant_name] = None

# ========== Rating session ==========
image_rows = df.sample(frac=1).reset_index(drop=True)  # Shuffle

for i, row in image_rows.iterrows():
    if pd.notna(row[participant_name]):
        continue  # Skip already-rated images

    image_path = os.path.join(row["Folder"], row["Image Name"])
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        continue

    # Resize image to fit within 800x800 window
    scale = min(800 / img.shape[1], 800 / img.shape[0])
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    cv2.imshow(f"Image {i + 1}/{len(df)} - {row['Image Name']}", resized_img)
    cv2.waitKey(1)

    key = None
    while key not in ["0", "1", "2"]:
        key = input("Rate this image [0 = clear, 1 = slight blur, 2 = blurred]: ").strip()

    df.at[i, participant_name] = int(key)
    cv2.destroyAllWindows()

# ========== Save results ==========
df.to_excel(excel_file, index=False)
print(f"\nAll ratings saved to: {excel_file}")
