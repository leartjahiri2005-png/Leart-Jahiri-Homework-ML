# Salient Object Detection (SOD) – Deep Learning Project

This project implements a deep learning model for Salient Object Detection using a custom CNN encoder-decoder architecture.  
The model takes an RGB image as input and predicts a binary mask highlighting the most visually important object in the image.

## Main Features
- Custom CNN encoder–decoder architecture (improved depth and filters)
- Binary cross-entropy + IoU training loss
- Train/validation split & full training loop implementation
- Metric tracking (IoU, Precision, Recall, F1, MAE)
- Automatic saving of best performing model
- Visual evaluation results and overlay predictions
- Inference demo on custom images

## Goals Achieved
- Built baseline SOD model
- Improved model with deeper layers & tuned learning rate
- Achieved significant performance improvement (Baseline → Improved)

## Datset
This project uses the ECSSD (Extended Complex Scene Saliency Dataset) containing 1000 natural images.
Dataset download link:
https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html


## How to Run
```bash
# Clone repository
git clone <your-repo-link>
cd Leart_Jahiri_SOD Project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python -m src.train

# Evaluate the model
python -m src.evaluate