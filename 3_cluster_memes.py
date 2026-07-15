import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os, shutil

data_file = 'processed_features.pkl'
clustered_file = 'clustered_dataset.pkl'
prev_folder = 'clustered_preview'
num_k = 12 

print("loading", data_file)
df = pd.read_pickle(data_file)

mat = np.stack(df['image_embedding'].values)

print(f"clustering into {num_k} groups...")
km = KMeans(n_clusters=num_k, random_state=42, n_init='auto')
df['cluster_id'] = km.fit_predict(mat)

df.to_pickle(clustered_file)
print("saved clustering output")

# clean old previews
if os.path.exists(prev_folder):
    shutil.rmtree(prev_folder) 

os.makedirs(prev_folder)

# make new folders for viewing
for i in range(num_k):
    cfolder = os.path.join(prev_folder, f"Template_{i}")
    os.makedirs(cfolder, exist_ok=True)
    
    # grab 3 examples
    egs = df[df['cluster_id'] == i].head(3)#pick first 3 images in template
    for _, r in egs.iterrows():
        source = os.path.join('meme_images', r['local_filename'])
        dest = os.path.join(cfolder, r['local_filename'])
        if os.path.exists(source):
            shutil.copy2(source, dest)

print("done grouping. check the preview dir.")