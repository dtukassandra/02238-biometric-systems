# Motion Blur Labeling for Biometric Image Quality

This project investigates the relationship between human-perceived motion blur and algorithmic quality assessment using OFIQ sharpness scores. It includes image annotation, sharpness scoring, and biometric utility evaluation through EDC, ROC, and DET curves, aligned with ISO/IEC 19795-1 and 29794-5.

---

## 📁 Project Structure

```
.
├── compute_ofiq_scores.py     # Compute OFIQ scores using ZeroMQ client
├── ofiq_zmq.py                # Patched OFIQ ZMQ server
├── rate_images.py             # GUI tool for human annotation of image blur
├── analyze_ratings.py         # Aggregates rater labels and agreement scores
├── main_analysis.py           # Full evaluation pipeline (EDC, ROC, DET)
├── metrics.py                 # Helper functions for metrics
├── image_ratings.xlsx         # Human rater annotations
├── ofiqlabels.csv             # OFIQ scores with blur label mapping
├── figures/                   # Output plots (EDC, ROC, DET, heatmaps)
```

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Additional requirements:
- Python ≥ 3.8
- OFIQ must be installed from [BSI-OFIQ/OFIQ-Project](https://github.com/BSI-OFIQ/OFIQ-Project)

---

### 2. Start the OFIQ Server

```bash
python ofiq_zmq.py
```

---

### 3. Score images with OFIQ

```bash
python compute_ofiq_scores.py --input_dir ./images --output_csv ofiqlabels_labeled.csv
```

---

### 4. Annotate Images (Human Blur Labels)

```bash
python rate_images.py --input_dir ./images
```

Ratings are saved to `image_ratings.xlsx`.

---

### 5. Analyze Ratings

```bash
python analyze_ratings.py
```

Generates inter-rater agreement stats and visual summaries.

---

### 6. Run Full Evaluation Pipeline

```bash
python main_analysis.py
```

Produces all figures including:
- EDC curves (OFIQ + Human)
- ROC and DET curves
- Heatmaps and label distribution
- Failure case contact sheet

---

## Modifications to `ofiq_zmq.py`

- `process_image(...)` method moved to class level for client access.
- `_ping()` method reinserted for proper server readiness check.
- `start()` indentation corrected.
- No changes made to message structure or algorithmic logic.

---

## License

This project is for academic use. OFIQ is licensed separately via Hochschule Darmstadt.

---

## Author

**Kassandra König**  
DTU MSc Student – 02238 Biometric Systems (Spring 2025)
