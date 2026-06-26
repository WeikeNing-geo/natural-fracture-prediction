markdown
# Deep Learning-Based Prediction and Modeling of Natural Fractures Within Ultra-Deep Reservoirs of Kuqa Depression, Tarim Basin of China

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the official MATLAB implementation of the models described in:

> **Weike Ning et al.** *Deep learning-based prediction and modeling of natural fractures within ultra-deep reservoirs of Kuqa Depression, Tarim Basin of China*  
> *Computers & Geosciences*, 2026 (submitted)

---

## Overview

We provide a complete computational workflow for intelligent natural fracture prediction using conventional well logs:

- **Fracture intensity prediction** – 1D Convolutional Neural Network (CNN)  
- **Fracture dip azimuth classification** – Bagged Tree ensemble (200 trees, 12 classes)  
- **Fracture dip angle classification** – Bagged Tree ensemble (200 trees, 6 classes)

All models accept **7 conventional well logs** as input.  
The default input features are named `Log_1` to `Log_7` (you can easily adjust these to match your own dataset’s column names, e.g., `DT`, `CAL`, `CNL`, `GR`, `DEN`, `RT`, `RM`).

The predictions serve as the basis for subsequent 3D discrete fracture network (DFN) modeling.

---

## Repository Structure
.
├── README.md
├── LICENSE
├── .gitignore
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
│ └── (generated prediction functions)
├── tests/
│ └── run_quick_test.m # One-click test of the entire pipeline
└── outputs/ # Created automatically – stores models and figures

text

---

## System Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+ recommended)
- **RAM**: Minimum 8 GB (16 GB recommended for full training with large datasets)
- **Disk Space**: ~500 MB for the code and example data; trained models may require up to 200 MB
- **GPU**: Optional but recommended for CNN training (NVIDIA GPU with CUDA support can greatly accelerate the process)

### Software Requirements

- **MATLAB** R2020a or later  
  *Required toolboxes*:  
  - **Deep Learning Toolbox**  
  - **Statistics and Machine Learning Toolbox**  
  - *Parallel Computing Toolbox* (optional – used automatically if available to speed up ensemble training)

All scripts are fully automatic – no interactive input is required. The random seed is fixed to ensure reproducibility.

---

## Quick Start (5–10 minutes)

1. **Clone the repository**
   ```bash
   git clone https://github.com/WeikeNing-geo/natural-fracture-prediction
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
Adjusting Feature Names
By default, the scripts expect the input feature columns to be named Log_1, Log_2, …, Log_7.
If your Excel file uses different names (e.g., DT, CAL, CNL, GR, DEN, RT, RM), simply modify the feature_names variable at the top of each training script:

matlab
feature_names = {'DT','CAL','CNL','GR','DEN','RT','RM'};
The same applies to the corresponding prediction functions.

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
All input Excel files must have 7 feature columns followed by 1 target column (for training) or just the 7 features (for prediction).
The default column names are Log_1 to Log_7 (features) and the last column is the target (e.g., FDI or class label).

Log_1	Log_2	Log_3	Log_4	Log_5	Log_6	Log_7	Target (if training)
...	...	...	...	...	...	...	...
For training: the target column must contain FDI (continuous value) for intensity, or integer class labels (1–12 for azimuth, 1–6 for dip angle).

For prediction: provide only the first 7 columns.

The example files in data/ follow this format exactly. If your column names differ, update the feature_names variable as described above.

Reproducing the Paper Results
Due to data confidentiality, the full dataset used in the paper cannot be shared. However:

All hyperparameters in the scripts are set exactly as described in the paper.

The provided example data retain the same structure and column names.

To obtain the exact figures and metrics reported, replace the example files with your own complete dataset (keeping the same column layout).

The random seed is fixed (rng(42) in all scripts), so the train/validation/test splits are deterministic.

License
This project is licensed under the MIT License – see the LICENSE file for details.

Contact
For questions regarding the code or the paper, please contact the corresponding author: Dr. Wei Ju.

