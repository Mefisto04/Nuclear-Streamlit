import numpy as np
import pandas as pd
import joblib
import os
import glob
from sklearn.preprocessing import MinMaxScaler
import json

# Load feature columns
with open('models/prediction/feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

print(f"Found {len(feature_columns)} feature columns: {feature_columns}")

# Find all CSV files
data_dir = 'NPPAD'
all_files = []
for root, _, _ in os.walk(data_dir):
    files = glob.glob(os.path.join(root, '*.csv'))
    all_files.extend(files)

print(f"Found {len(all_files)} CSV files")

# Sample a subset of files for creating the scaler
sample_size = min(20, len(all_files))
sampled_files = all_files[:sample_size]

# Collect feature data
all_data = []
for file in sampled_files:
    try:
        df = pd.read_csv(file)
        
        # Check if necessary columns exist
        available_cols = [col for col in feature_columns if col in df.columns]
        if len(available_cols) < 3:  # Require at least 3 features
            print(f"Warning: File {file} has too few columns ({len(available_cols)})")
            continue
            
        # Extract features that are available
        data = df[available_cols].values
        all_data.append(data)
        print(f"Processed {file} with {len(available_cols)} features")
    except Exception as e:
        print(f"Error processing {file}: {e}")
        continue

# Combine all data
if all_data:
    combined_data = np.vstack(all_data)
    print(f"Combined data shape: {combined_data.shape}")
    
    # Create and fit the scaler
    scaler = MinMaxScaler()
    scaler.fit(combined_data)
    
    # Save the scaler
    os.makedirs('models/prediction', exist_ok=True)
    joblib.dump(scaler, 'models/prediction/scaler.pkl')
    print("Scaler created and saved to models/prediction/scaler.pkl")
    
    # Save the actual feature columns used
    with open('models/prediction/feature_columns.json', 'w') as f:
        json.dump(available_cols, f, indent=4)
    print(f"Updated feature_columns.json with {len(available_cols)} available features")
else:
    print("No data collected. Cannot create scaler.") 