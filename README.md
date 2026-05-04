# Business-Forecasting-Semester-Project
# Detecting Baseballs in Training Videos
## An Object Detection Approach Using Faster R-CNN

**Authors:** Austin Barrette and Aman Jayswal
**Course:** Econ 8310 - Business Forecasting (Spring 2026)
**Instructor:** Dr. Dustin White

---

## Project Overview

This project trains a Faster R-CNN object detection model to identify baseballs in coaching training videos. Given short clips of pitching and hitting drills, the model draws bounding boxes around every visible baseball in each frame, with the goal of helping coaches quickly locate the moments that matter (swing, contact, ball flight) without scrubbing through long stretches of empty footage.

**Final Test Set Performance:**

| Metric | Value |
|---|---|
| mAP @ IoU=0.50 | **0.83** |
| mAP @ IoU=0.50–0.95 | 0.60 |
| Mean IoU (correct detections) | 0.86 |
| Precision | 0.82 |
| Recall | 0.84 |
| F1 Score | 0.83 |

For full methodology, results discussion, and limitations, see the [Executive Summary](Executive_Summary_Final.docx).

---

## Repository Contents

```
.
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── Executive_Summary_Final.docx           # Final project write-up
├── baseball_detection_training.ipynb      # Main training notebook
├── training_curves.png                    # Loss curves visualization
├── evaluation_results.json                # Test set metrics
└── annotations/                           # 38 XML annotation files (CVAT format)
```

**Files NOT in this repository (available in Google Drive — see link below):**

- `videos/` — 38 raw .mov files used for training (too large for GitHub)
- `baseball_model_best.pt` — trained model weights, best validation epoch (~150 MB)
- `baseball_model_final.pt` — trained model weights, final epoch (~150 MB)

### Google Drive Link

All large files (videos and trained model weights) are available here:
**[INSERT GOOGLE DRIVE LINK]**

---

## How to Run This Project

The training pipeline is built as a single Google Colab notebook that runs end-to-end on a free T4 GPU in approximately one hour.

### Step 1: Get the data

1. Download the `videos/` folder from the [Google Drive link](#google-drive-link) above
2. Clone this repository to get the `annotations/` folder
3. Upload both folders to your own Google Drive in a single parent folder (e.g. `My Drive/Econ8310_Project/`)

Your Drive folder structure should look like this:

```
My Drive/
└── Econ8310_Project/
    ├── videos/         # 38 .mov files
    └── annotations/    # 38 .xml files
```

### Step 2: Open the notebook in Colab

1. Open `baseball_detection_training.ipynb` in Google Colab
2. Set the runtime: **Runtime → Change runtime type → T4 GPU → Save**
3. In Cell 1, update the `PROJECT_DIR` path if your folder is named something other than `Econ8310_Project`

### Step 3: Run all cells in order

| Cell | What it does | Approximate time |
|---|---|---|
| 1 | Mount Google Drive and verify file counts | <1 min |
| 2 | Verify GPU is active | <1 min |
| 3 | Build the data loader, apply quality filters, split 70/15/15 | 2–4 min |
| 4 | Load Faster R-CNN with COCO pretrained weights | <1 min |
| 5 | Train for 10 epochs, save best model by validation loss | ~60 min |
| 6 | Plot training curves and save to Drive | <1 min |
| 7 | Inference sanity check on a sample video | <1 min |
| 8 | Evaluate on held-out test set (mAP, IoU, precision, recall, F1) | 2–3 min |

Trained model weights and result files save automatically to your `Econ8310_Project/` folder in Drive.

---

## Data Pipeline Notes

The data loader applies two quality filters to the 60 raw class-shared annotation files, leaving 38 high-quality video/XML pairs for training:

- **Minimum 10 annotated frames per video** — excludes incomplete annotations
- **At least one moving ball annotation per video** — excludes videos that were not fully annotated for motion

The `moving` attribute is read from the CVAT XML as a child element (`<attribute name="moving">true</attribute>`), not as a direct box attribute.

The full dataset breakdown:

- 38 videos
- ~2,450 annotated frames
- ~22,500 baseball annotations (1,328 moving + ~21,200 stationary)

---

## Model Architecture

We used **Faster R-CNN with a ResNet-50 FPN backbone**, pretrained on the COCO dataset, and applied transfer learning. The pretrained feature extraction layers were preserved, and only the final classification head was replaced to predict two classes: background and baseball.

Training hyperparameters:

- **Optimizer:** Stochastic Gradient Descent with momentum 0.9
- **Learning rate:** 0.001 (reduced from default 0.005 to prevent NaN loss)
- **Scheduler:** StepLR, gamma 0.1, step size 3 epochs
- **Batch size:** 2 (constrained by GPU memory)
- **Input size:** 224 × 224 pixels
- **Epochs:** 10
- **Best epoch:** 8 (validation loss 0.2281)

---

## Acknowledgments

This project was built on guidance from Dr. Dustin White throughout the semester:

- The recommendation to use a pretrained PyTorch object detection model (proposal feedback)
- The recommendation to use Colab's free GPU rather than local CPU (HW3 Discord)
- The recommendation to load all data upfront for projects of this size (HW3 Discord)
- The provided starter code structure shared after Assignment 3

We also thank our classmates whose Discord discussions surfaced the NaN loss issue at LR=0.005, which informed our LR=0.001 choice.

---

## Requirements

See [`requirements.txt`](requirements.txt). All packages are pre-installed on Google Colab except `torchmetrics`, which the notebook installs via `pip` automatically.
