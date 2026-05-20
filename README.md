# Laptop Price Prediction & Market Analysis (Algeria)

## Overview
This project focuses on cleaning, analyzing, and modeling data collected from Algerian online laptop markets.

The dataset was obtained through web scraping and contains laptop specifications such as CPU, GPU, RAM, storage, screen characteristics, and price.

Main objectives:
- Clean and standardize laptop specifications
- Predict laptop prices using regression models
- Group similar laptops using clustering
- Explore associations between hardware components

---

## Dataset Columns

- `price_preview`
- `created_at`
- `city`
- `spec_Etat`
- `model_name`
- `cpu_name`
- `cores`
- `cpu_mark`
- `cpu_tdp`
- `gpu_name`
- `g3d_mark`
- `g2d_mark`
- `gpu_tdp`
- `RAM_SIZE`
- `SSD_SIZE`
- `HDD_SIZE`
- `SCREEN_SIZE`
- `SCREEN_FREQUENCY`
- `SCREEN_RESOLUTION`
- `RAM_TYPE`

---

## Notebook Execution Sequence

Run the notebooks in this order to process the data correctly:

1. **`notebooks/feature-engineering/preProcessing.ipynb`**
   - Input: `data/raw/data.csv`
   - Output: `data/processed/data_cleaned.csv`
   - Tasks: Remove duplicates, handle missing values, general data cleaning

2. **`notebooks/feature-engineering/cpus_gpus_handling.ipynb`**
   - Input: `data/processed/data_cleaned.csv`, `data/raw/cpus.csv`, `data/raw/gpus.csv`
   - Output: `data/processed/data_with_cpus_gpus.csv`
   - Tasks: Normalize and merge CPU/GPU information

3. **`notebooks/cleaning/clean_ram_storage.ipynb`**
   - Input: `data/processed/data_with_cpus_gpus.csv`
   - Output: `data/processed/cleanedramstoragedata.csv`
   - Tasks: Standardize RAM types and storage configurations using mappings

4. **`notebooks/cleaning/clean_price.ipynb`**
   - Input: Cleaned dataset from step 3
   - Output: Dataset with corrected prices
   - Tasks: Detect and remove troll prices, validate price ranges

5. **`notebooks/cleaning/clean_screen_related.ipynb`**
   - Input: Dataset from step 4
   - Output: Dataset with standardized screen specifications
   - Tasks: Normalize screen sizes and resolutions

6. **`notebooks/association/association_rules.ipynb`** (optional)
   - Input: Fully cleaned dataset
   - Output: Association rules between hardware components
   - Tasks: Explore hardware component correlations

## Project Tasks

### Data Cleaning
- Standardize screen sizes and resolutions
- Handle missing values using:
  - model-specific mode
  - rule-based inference
  - limited ML-based imputation
- Normalize CPU/GPU information using benchmark scores
- Detect and correct abnormal prices

### Feature Engineering
- Resolution categorization (`HD`, `FHD`, `QHD`, `4K`, ...)
- Canonical screen sizes
- Hardware performance indicators
- Ordinal encoding for some specifications

### Price Prediction
Regression models are used to estimate laptop prices from technical specifications.

Models:
- Linear Regression
- Random Forest
- XGBoost

### Clustering
Group laptops into categories such as:
- Gaming
- Student
- Business
- High-end workstation

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook / Google Colab

---


## Notes
- Some missing values are intentionally kept when confidence is low.
- Resolution and screen-size normalization follow common industry standards.
- The dataset may contain noisy or inconsistent scraped values.
