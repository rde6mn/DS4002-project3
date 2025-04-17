# Power of Machine Learning in Distinguishing Skin Lesions

## Repository Overview
Welcome to the Data Dinosaurs' project repository! This repository contains our analysis of skin lesions from the ISIC Skin Cancer Dataset as part of our **DS 4002 - Data Science Project**. The project explores whether a machine learning-based classification model can distinguish between malignant and benign skin lesions from the dataset utilizing **image data in Python with a Random Forest Classifier** and achieve at least 90% accuracy. 

## Team Members
- **Amani Akkoub** (ctf3un) - Group Leader  
- **Mohini Gupta** (rde6mn) 
- **Neha Dacherla** (mcn4ws)  

## Course Details
- **Course**: DS 4002 - Data Science Project  
- **Date**: April 17, 2025  

## Hypothesis
A Random Forest classifier, trained on preprocessed features from the ISIC Skin Cancer Dataset, can achieve at least 90% classification accuracy, with precision, recall, and F1-scores indicating robust performance across various classification scenarios.

## Research Question
Can a machine learning model trained on the ISIC Skin Cancer Dataset achieve dermatologist-level accuracy in distinguishing malignant from benign skin lesions based on existing studies?

## Data Source and Preprocessing
 **Source**: International Skin Imaging Collaboration (ISIC) 
- **Sites**:  
  - PROVe-AI: 603 lesions images from 435 patients 
- **Time Frame**: November 2022

### Preprocessing Steps & Feature Engineering:
- Resized all images to 224x224 pixels and converted them to grayscale
- Missing or corrupted images identified and removed to ensure data quality
- Extracted features using either Principal Component Analysis (PCA) or Histogram of Oriented Gradients (HOG)
- Experimented with both methods and selected the HOG because it yielded better model performance
- Preprocessed data was split into training (70%), validation (15%), and test (15%) sets

## Exploratory Data Analysis (EDA)
- **Visualized images data** 
- Age Distribution of ISIC Archive displays left skew showing that data collection of skin lesions is from **older patients**
- Pie Chart of Benign vs. Malignant Count in ISIC Archive illustrates **greater benign lesions** than malignant

## Model Approach
We use **Python** to conduct image data analysis due to its robust modeling and identification capabilities. The analysis was performed on the Windows platform. Our workflow includes:

**Visualizing Metadata**: We explored variables such as sex, age, and personal history of melanoma to understand their distribution and relationship to lesion type.

**Extracting Image Features**: Instead of raw pixel data, we used color histograms (Red, Green, and Blue channel distributions) to summarize visual features of each lesion image.

**Data Preprocessing**: We handled missing values, encoded categorical variables, and merged metadata with extracted image features for modeling.

**Logistic Regression Classifier**: We leveraged Scikit-Learn in Python to train a logistic regression model that classifies lesions as benign or malignant.

**Model Evaluation**: Performance was assessed using accuracy, confusion matrix, and precision/recall metrics, focusing on identifying benign lesions with high confidence.

## Methodology [NEED TO DO]
We used the ISIC skin lesion dataset consisting of over 600 dermoscopic images and an associated metadata CSV file.

**Image Processing**: Each image was opened using the PIL library and converted to RGB. We computed color histograms for the Red, Green, and Blue channels (256 bins each), resulting in a shape of (3, 256) for each image’s histogram.

**Metadata Integration**: Metadata was cleaned and merged with the image data using the ISIC ID as a key. Features such as image size, patient sex, age, lesion location, and personal history of melanoma were included.

**Data Preprocessing**: Non-numeric variables (e.g., 'sex', 'anatom_site_general') were encoded using OneHotEncoding. Missing data was handled where necessary. Target variable was 'benign_malignant' (binary: benign vs malignant).

**Exploratory Data Analysis (EDA)**: We visualized the average color histograms for benign vs malignant lesions to observe differences in color profiles. We examined distribution of metadata fields (sex, age, personal history).

**Modeling**: We used a Logistic Regression classifier (Scikit-Learn) to model lesion classification. Histogram values and metadata were used as features. The data was split into training (80%) and test (20%) sets.

**Evaluation**: We evaluated performance using a confusion matrix and classification report. The model achieved high accuracy (~79%) in identifying benign lesions, though performance on malignant lesions was lower.

## Reproduceability [NEED TO DO]
To reproduce our analysis:

Clone or download the dataset folder (images) and the metadata.csv file.

Make sure all image files are in a folder named ISIC-images in your working directory.

Run the Python script (SkinModel.py) containing all processing, modeling, and evaluation steps.

The script is deterministic (no randomness unless random_state is removed), ensuring reproducibility of results.

Output includes visualizations and classification metrics printed to the console and/or displayed with matplotlib.

# Required Python Packages [NEED TO ADD RANDOM FOREST AND OTHER LIBRARIES POST ANALYSIS)
Ensure the following Python packages are installed before running the code:

import os               # For navigating file system

import pandas as pd     # For metadata handling and DataFrame manipulation

from PIL import Image   # For image loading and conversion

import numpy as np      # For numerical computations

import matplotlib.pyplot as plt  # For visualizations

from sklearn.model_selection import train_test_split  # For data splitting

from sklearn.linear_model import LogisticRegression   # For classification

from sklearn.metrics import confusion_matrix, classification_report  # For evaluation

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

## Goal
To analyze skin lesions from the ISIC dataset to achieve at least 90% classification accuracy while evaluating precision, recall, and F1-scores to ensure robust performance. 

## Repo Structure [MIGHT NEED TO EDIT]
├── data/
│   ├── metadata.csv
├── notebooks/
│   ├── exploratory_analysis.python
│   ├── modeling.python
├── results/
│   ├── plots/
│   ├── classification/
|---data appendix
├── README.md

## Conclusion
Our analysis aims to provide insights into the distribution and characteristics of skin lesions within the ISIC dataset. By exploring imaging patterns, we hope to highlight the interplay between biological factors, lesion location, and diagnosis trends in the context of skin cancer detection. For any questions or contributions, please reach out to the team!

Data Dinosaurs | DS 4002 - Data Science Project

