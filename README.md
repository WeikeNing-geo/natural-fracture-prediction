markdown
# Deep Learning-Based Prediction of Natural Fractures in Ultra-Deep Reservoirs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the official MATLAB implementation of the models described in:

> **[Your Paper Title]**  
> *Computers & Geosciences*, 2025

## Overview

We provide a complete workflow for intelligent natural fracture prediction using conventional well logs:

- **Fracture intensity prediction** – 1D Convolutional Neural Network (CNN)  
- **Fracture dip azimuth classification** – Bagged Tree ensemble (200 trees, 12 classes)  
- **Fracture dip angle classification** – Bagged Tree ensemble (200 trees, 6 classes)

All models accept **7 conventional well logs** as input:  
`DT, CAL, CNL, GR, DEN, RT, RM`

The predictions serve as the basis for subsequent 3D discrete fracture network (DFN) modeling.

---

## Repository Structure
.
├── README.md
├── LICENSE
├── data/
│ ├── Date_Fracture_intensity.xlsx # Example training data for CNN (FDI)
│ ├── Date_Fracture_orientation_Dip_azimuth.xlsx # Example training data for dip azimuth
│ ├── Date_Fracture_orientation_Dip_angle.xlsx # Example training data for dip angle
│ ├── DYC.xlsx # Example new well for intensity prediction
│ └── new_well.xlsx # Example new well for orientation prediction
├── src/
│ ├── cnn_fdi_main_.m # CNN training script (fracture intensity)
│ ├── predict_fdi.m # Standalone FDI prediction function
│ ├── bagged_trees_dip_azimuth_main.m # Training script for dip azimuth
│ ├── bagged_trees_dip_angle_main.m # Training script for dip angle
│ └── predict_dip_azimuth.m # Generated after training azimuth model
│ └── predict_dip_angle.m # Generated after training dip angle model
├── tests/
│ └── run_quick_test.m # One-click test of the entire pipeline
└── outputs/ # Created automatically – stores models and figures

text

---

## Requirements

- **MATLAB** R2020a or later
- **Toolboxes**:
  - Deep Learning Toolbox
  - Statistics and Machine Learning Toolbox
  - *Parallel Computing Toolbox (optional – used if available to speed up training)*

All scripts are **fully automatic** – no interactive input is required. The random seed is fixed to ensure reproducibility.

---

## Quick Start (5–10 minutes)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
Open MATLAB and navigate to the repository folder.

Run the quick test

matlab
run('tests/run_quick_test.m')
This script will:

Train the CNN on the example intensity data (with reduced epochs for speed)

Train the Bagged Trees classifiers for dip azimuth and dip angle

Generate evaluation figures and save them in outputs/

Print performance metrics (RMSE, R², confusion matrices) in the command window

Note: The example datasets are small and anonymized; they are intended only to verify that the code runs correctly. Full training results reported in the paper require the complete (confidential) dataset.

How to Use with Your Own Data
1. Train the CNN for Fracture Intensity
Edit the data_file variable at the top of cnn_fdi_main_.m to point to your Excel file, then run:

matlab
cnn_fdi_main_
The script will output:

outputs/cnn_model.mat – trained model and normalization parameters

outputs/evaluation_metrics.csv – RMSE and R² for train/val/test sets

Various figures (metrics bar chart, scatter plots, loss curve, residual histogram)

2. Train the Bagged Trees Classifiers for Orientation
Similarly, modify the data_file variable in:

matlab
bagged_trees_dip_azimuth_main
bagged_trees_dip_angle_main
These will generate:

results_azimuth/bagged_trees_azimuth.mat and results_dip_angle/bagged_trees_dip_angle.mat

Standalone prediction functions predict_dip_azimuth.m and predict_dip_angle.m

Confusion matrices, TPR matrices, and feature importance plots.

3. Predict on a New Well
Once a model has been trained, use the corresponding prediction function:

matlab
% Fracture intensity
pred_fdi = predict_fdi('new_well_data.xlsx', 'outputs/cnn_model.mat', 'predicted_fdi.csv');

% Dip azimuth
pred_az = predict_dip_azimuth('new_well_data.xlsx');

% Dip angle
pred_ang = predict_dip_angle('new_well_data.xlsx');
The new well file must contain only the 7 feature columns (same order and column names as the training data). Refer to the example files in data/.

Data Format
All input Excel files must have the following columns (case-sensitive, order matters):

DT	CAL	CNL	GR	DEN	RT	RM	FDI (or Class)
For training: include an 8th column containing the target (FDI for intensity, or integer class label 1–12 for azimuth, 1–6 for dip angle).

For prediction: provide only the first 7 columns.

The example files in data/ follow this format exactly. If your column headers differ, modify the feature_names variable at the top of each training script.

Reproducing the Paper Results
Due to data confidentiality, the full dataset used in the paper cannot be shared. However:

All hyperparameters in the scripts are set exactly as described in the paper.

The provided example data retain the same structure and column names.

To obtain the exact figures and metrics reported, replace the example files with your own complete dataset (keeping the same column layout).

The random seed is fixed (rng(42) in all scripts), so the train/validation/test splits are deterministic.

License
This project is licensed under the MIT License – see the LICENSE file for details.

Citation
If you use this code in your research, please cite:

text
@article{YourPaper2025,
  author    = {Your Name and Co-authors},
  title     = {Deep learning-based prediction and modeling of natural fractures within ultra-deep reservoirs of Kuqa Depression, Tarim Basin of China},
  journal   = {Computers & Geosciences},
  year      = {2025},
  doi       = {your-doi}
}
Contact
For questions regarding the code or the paper, please contact the corresponding author: Dr. Wei Ju.

