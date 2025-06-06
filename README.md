# Yerba Mate Phenolic Extraction Analysis

## Summary
- [Yerba Mate Phenolic Extraction Analysis](#yerba-mate-phenolic-extraction-analysis)
  - [Summary](#summary)
  - [Introduction](#introduction)
  - [Objective](#objective)
  - [Installation \& Setup](#installation--setup)
  - [Repository Structure](#repository-structure)
    - [Color Extraction colorsExtraction](#color-extraction-colorsextraction)
    - [Data Analysis dataAnalysis](#data-analysis-dataanalysis)
    - [Implementation implementation.py](#implementation-implementationpy)
  - [Model Performance](#model-performance)
  - [Data Requirements](#data-requirements)

## Introduction
Ilex paraguariensis contains numerous bioactive compounds and is a form of social interaction in many countries. Yerba mate beverages generally contain higher total polyphenol content (TPC) compared to other plant-based drinks, with hot water-extracted mate offering a higher amount of polyphenol intake, highlighting its importance as a source of antioxidants. For TPC quantification, the colorimetric method using smartphones combined with the Folin-Ciocalteu (FC) assay has been employed for a low-cost and rapid evaluation in different products. However, this approach relies on the FC assay, limiting this method’s accessibility due to the reagent costs and equipment involved.

The study of this repository is present at this [link]().

## Objective

This repository provides tools for analyzing and predicting Total Polyphenol Content (TPC) in yerba mate aqueous extracts before a Folin-Ciocalteu assay using image processing and machine learning. The system uses RGB and HSV color features along with pH measurements to estimate phenolic compound concentrations.

## Installation & Setup

1. Clone the repository:
 
```sh
git clone https://github.com/GustavoSantiago113/YerbaMatePhenolicExtraction.git
cd YerbaMatePhenolicExtraction
```

2. Create and activate a virtual environment (recommended):

```sh
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

3. Install dependencies:

```sh
pip install -r requirements.txt
```

## Repository Structure

### Color Extraction [colorsExtraction](colorsExtraction/)

Tool for extracting color features from yerba mate extract images obtained using a cellphone camera and the [image capture chamber](https://data.mendeley.com/datasets/5rdnmmbnjk/1).

**Usage:**
1. Place your images in the "images" folder
2. Run the color extraction script:

```python
python colorsExtraction/colorExtraction.py
```

3. Follow the GUI prompts:
   * Select folder containing images
   * Click on points of interest in each image
   * Confirm selection using the button
4. After iterating over all images of the foler, the results will be saved in colorsExtraction/color_points.csv

### Data Analysis [dataAnalysis](dataAnalysis/)

Contains scripts in R and Python for model training and evaluation.

**Components:**

* data-analysis.R: R script for initial data analysis, feature selections with XGB-Boruta and Pearson Correlation and visualization.

* modelsWithOptuna.py: Implements and optimizes multiple machine learning models using Optuna:
    * Gradient Boosting
    * XGBoost
    * Elastic Net
    * SVM
    * Polynomial Regression
    * A simple Neural Network
  
* modelExport.py: Exports the best performing model (Elastic Net) for implementation.
  
* Output files:
    * model_metrics_optuna.csv: Performance metrics for all models.
    * best_hyperparameters.txt: Optimized parameters for each model.
    * model_comparisons_optuna.png: Visual comparison of model performances.

### Implementation [implementation.py](implementation.py)

Interactive tool for predicting TPC from new images using the trained model.

**Usage:**
1. Run the implementation script:

```python
python implementation.py
```
2. Follow the interactive steps:
    * Select an image when prompted
    * Click on a point in the image to analyze
    * Enter the pH value in the text box
    * Click "Confirm" to get the TPC prediction

## Model Performance

The implemented Elastic Net model achieves:

* RMSE: 128.53
* RRMSE: 40.66%
* KGE: 0.696
* R²: 0.612

## Data Requirements

* Input images should be well-lit photographs of yerba mate extract inside the cuvette within the [image capture chamber](https://data.mendeley.com/datasets/5rdnmmbnjk/1).
* pH measurements.
* Supported image formats: .jpg, .jpeg, .png, .bmp