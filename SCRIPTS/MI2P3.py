

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('metadata.csv')

# Set seaborn theme
sns.set(style="whitegrid")

# -----------------------
# 1. Basic Overview
# -----------------------
print(df.info())
print(df.describe(include='all'))

# -----------------------
# 2. Age Distribution
# -----------------------
plt.figure(figsize=(8, 5))
sns.histplot(df['age_approx'], bins=18, kde=True)
plt.title('Age Distribution')
plt.xlabel('Approximate Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# -----------------------
# 3. Sex Distribution
# -----------------------
plt.figure(figsize=(6, 4))
sns.countplot(x='sex', data=df, palette='Set2')
plt.title('Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# -----------------------
# 4. Anatomical Site Distribution
# -----------------------
plt.figure(figsize=(10, 5))
sns.countplot(y='anatom_site_general', data=df, order=df['anatom_site_general'].value_counts().index, palette='coolwarm')
plt.title('Lesion Anatomical Location')
plt.xlabel('Count')
plt.ylabel('Anatomical Site')
plt.tight_layout()
plt.show()

# -----------------------
# 5. Diagnosis Distribution INCLUDE THIS
# -----------------------
plt.figure(figsize=(10, 5))
sns.countplot(y='diagnosis_1', data=df, order=df['diagnosis_1'].value_counts().index, palette='viridis')
plt.title('Diagnosis Distribution')
plt.xlabel('Count')
plt.ylabel('Diagnosis Category')
plt.tight_layout()
plt.show()

# -----------------------
# 6. Diagnosis Confirmation Type
# -----------------------
plt.figure(figsize=(6, 4))
sns.countplot(x='diagnosis_confirm_type', data=df, palette='pastel')
plt.title('Diagnosis Confirmation Method')
plt.xlabel('Confirmation Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# -----------------------
# 7. Fitzpatrick Skin Type
# -----------------------
if 'Fitzpatrick_skin_type' in df.columns:
    plt.figure(figsize=(7, 4))
    sns.countplot(x='Fitzpatrick_skin_type', data=df, palette='muted', order=sorted(df['Fitzpatrick_skin_type'].dropna().unique()))
    plt.title('Fitzpatrick Skin Type Distribution')
    plt.xlabel('Skin Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
