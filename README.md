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
## Reproduceability [NEED TO DO]

# Required Python Packages [NEED TO ADD RANDOM FOREST AND OTHER LIBRARIES POST ANALYSIS)
Ensure the following Python packages are installed before running the code:

library(pandas)
library(matplotlib)
library(seaborn)

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

