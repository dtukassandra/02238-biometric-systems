"""
compute_ofiq_scores.py

Scores all images in a directory using the OFIQ ZMQ server and saves the resulting sharpness values.

Inputs:
- Image files (jpg, png, bmp, tif...) located anywhere under the current directory

Requirements:
- OFIQ ZMQ server executable (e.g., compiled OFIQ_zmq_app)
- Patched ofiq_zmq.py client with `process_image(...)` exposed

Outputs:
- ofiqlabels.csv : CSV file with image names, folders, and OFIQ sharpness scores

Usage:
- Set the correct path to the OFIQ executable in `ofiq_executable`
- Place images under the specified `image_root` directory
- Run the script to score all images and write output to disk

Dependencies:
- pandas, pathlib
"""

from ofiq_zmq import OfiqZmq, OfiqQualityMeasure
from pathlib import Path
import pandas as pd

# ========== Configuration ==========
ofiq_executable = Path("/path/to/OFIQ_zmq_app")
image_root = Path(".")
output_file = "ofiqlabels.csv"

# ========== Start OFIQ client ==========
ofiq = OfiqZmq(str(ofiq_executable))

# Collect all supported image formats
image_paths = []
for ext in ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]:
    image_paths += list(image_root.rglob(f"*.{ext}")) + list(image_root.rglob(f"*.{ext.upper()}"))

print(f"Found {len(image_paths)} images.")

results = []
for idx, image_path in enumerate(image_paths, 1):
    print(f"Processing image {idx}/{len(image_paths)}: {image_path.name}")
    res = ofiq.process_image(image_path)
    score = res["quality_assessments"][OfiqQualityMeasure.SHARPNESS].scalar_score if res else -1
    results.append({
        "Image Name": image_path.name,
        "Folder": image_path.parent.name,
        "OFIQ Sharpness": score
    })

ofiq.shutdown()

# ========== Save results ==========
df = pd.DataFrame(results)
# Binarize OFIQ scores: 0 = clear, 1 = blurred
df["OFIQ Sharpness Label"] = (df["OFIQ Sharpness"] < 30).astype(int)

df.to_csv(output_file, index=False)
print(f"Done. Results saved to: {output_file}")
