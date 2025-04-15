# Power of Machine Learning in Distinguishing Skin Lesions

## Repository Overview
Welcome to the Data Dinosaurs' project repository! This repository contains our analysis of skin lesions from the ISIC Skin Cancer Dataset as part of our **DS 4002 - Data Science Project**. The project explores whether a machine learning-based classification model can distinguish between malignant and benign skin lesions from the dataset utilizing **image data in R with a Random Forest Classifier** and achieve at least 90% accuracy. 

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



