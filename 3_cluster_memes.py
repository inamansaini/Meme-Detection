import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
import shutil

INPUT_FILENAME = 'processed_features.pkl'
OUTPUT_FILENAME = 'clustered_dataset.pkl'
PREVIEW_DIR = 'clustered_preview'
NUMBER_OF_CLUSTERS = 25 # Kept smaller for API scraping

print(f"Loading {INPUT_FILENAME}...")
df = pd.read_pickle(INPUT_FILENAME)

X = np.stack(df['image_embedding'].values)

print(f"Running K-Means to find {NUMBER_OF_CLUSTERS} templates...")
kmeans = KMeans(n_clusters=NUMBER_OF_CLUSTERS, random_state=42, n_init='auto')
df['cluster_id'] = kmeans.fit_predict(X)

df.to_pickle(OUTPUT_FILENAME)
print(f"Saved to {OUTPUT_FILENAME}")

# Create previews
if not os.path.exists(PREVIEW_DIR):
    os.makedirs(PREVIEW_DIR)

for cluster_num in range(NUMBER_OF_CLUSTERS):
    cluster_folder = os.path.join(PREVIEW_DIR, f"Template_{cluster_num}")
    os.makedirs(cluster_folder, exist_ok=True)
    
    samples = df[df['cluster_id'] == cluster_num].head(3)
    for _, row in samples.iterrows():
        src = os.path.join('meme_images', row['local_filename'])
        dst = os.path.join(cluster_folder, row['local_filename'])
        if os.path.exists(src):
            shutil.copy2(src, dst)

print(f"Check the '{PREVIEW_DIR}' folder to see the groupings.")