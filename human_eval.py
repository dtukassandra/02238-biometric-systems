import os
import random
import pandas as pd
import cv2
from openpyxl import load_workbook

# Configuration
image_dir = os.getcwd()
subfolders = [f.name for f in os.scandir(image_dir) if f.is_dir()]
excel_file = "human_eval.xlsx"
participant_name = input("Enter your name or ID (e.g., rater1): ")

# Load existing or create new DataFrame
if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    # Build image list from all subfolders
    image_list = []
    for subfolder in subfolders:
        for fname in os.listdir(subfolder):
            if fname.lower().endswith(".jpg"):
                image_list.append({
                    "Image Name": fname,
                    "Folder": subfolder
                })
    df = pd.DataFrame(image_list)

# Skip if column already exists
if participant_name in df.columns:
    print(f"Participant '{participant_name}' already has a column. Overwriting values.")
else:
    df[participant_name] = None

# Shuffle order
image_rows = df.sample(frac=1).reset_index(drop=True)

# Prompt user for rating
for i, row in image_rows.iterrows():
    image_path = os.path.join(row["Folder"], row["Image Name"])
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        continue

    # Resize image to fit screen (e.g., width 800 while keeping aspect ratio)
    scale_width = 800 / img.shape[1]
    scale_height = 800 / img.shape[0]
    scale = min(scale_width, scale_height)

    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    cv2.imshow(f"Image {i + 1}/{len(df)}: Rate this image (0 = clear, 1 = slight blur, 2 = blurred)", resized_img)
    cv2.waitKey(1)  # required for some platforms to show window
    key = None

    while key not in ["0", "1", "2"]:
        key = input(f"Enter rating for {row['Image Name']} [0=clear, 1=slight blur, 2=blurred]: ")

    df.at[i, participant_name] = int(key)
    cv2.destroyAllWindows()
    df.to_excel(excel_file, index=False)  # Save after each rating

# Save updated file
df.to_excel(excel_file, index=False)
print(f"Evaluation saved in: {excel_file}")
