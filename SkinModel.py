import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Paths
folder_path = 'ISIC-images'
metadata_path = 'metadata.csv'

# Load metadata
metadata_df = pd.read_csv(metadata_path)

# Convert metadata to dictionary for quick access
metadata_df['isic_id'] = metadata_df['isic_id'].astype(str)
metadata_dict = metadata_df.set_index('isic_id').to_dict(orient='index')

# Initialize list to store combined image+metadata info
image_data = []

# Process each image
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.jpg'):
        isic_id = filename.replace('.jpg', '')  # strip extension to get the id
        path = os.path.join(folder_path, filename)

        if isic_id not in metadata_dict:
            continue  # skip if no metadata for this image

        with Image.open(path) as img:
            img = img.convert('RGB')
            size = img.size  # (width, height)
            hist = np.array(img.histogram()).reshape(3, 256)

            # Combine image info with metadata
            entry = {
                'isic_id': isic_id,
                'width': size[0],
                'height': size[1],
                'histogram': hist,
                **metadata_dict[isic_id]  # expand metadata fields
            }
            image_data.append(entry)

# Convert to DataFrame for analysis (excluding histogram for now)
df = pd.DataFrame([{k: v for k, v in item.items() if k != 'histogram'} for item in image_data])

print(f"Total images with metadata: {len(df)}")
print(df.head())

benign = [item['histogram'] for item in image_data if item['benign_malignant'] == 'benign']
malignant = [item['histogram'] for item in image_data if item['benign_malignant'] == 'malignant']

avg_benign = np.mean(benign, axis=0)
avg_malignant = np.mean(malignant, axis=0)

plt.figure(figsize=(10, 4))
for i, color in enumerate(['r', 'g', 'b']):
    plt.plot(avg_benign[i], color=color, label=f'Benign - {color.upper()}', linestyle='--')
    plt.plot(avg_malignant[i], color=color, label=f'Malignant - {color.upper()}')

plt.title('Average Color Histograms: Benign vs Malignant')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Step 1: Build a DataFrame that includes histogram features
flat_data = []
labels = []
metadata_features = []

for item in image_data:
    if item['benign_malignant'] in ['benign', 'malignant']:
        flat_hist = item['histogram'].flatten()  # Flatten RGB histograms (3x256 = 768)
        flat_data.append(flat_hist)
        labels.append(1 if item['benign_malignant'] == 'malignant' else 0)
        metadata_features.append({
            'age_approx': item.get('age_approx'),
            'sex': item.get('sex'),
            'anatom_site_general': item.get('anatom_site_general')
        })

X_hist = pd.DataFrame(flat_data)
X_meta = pd.DataFrame(metadata_features)
y = pd.Series(labels, name='target')

# Step 2: Preprocessing
numeric_features = ['age_approx']
categorical_features = ['sex', 'anatom_site_general']

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])

# Combine image + metadata
X_combined = pd.concat([X_hist.reset_index(drop=True), X_meta.reset_index(drop=True)], axis=1)

# Fix: convert all column names to strings
X_combined.columns = X_combined.columns.astype(str)


# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Build pipeline with classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Step 5: Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
