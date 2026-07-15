import os
# Fix for WinError 1114: Prevents OpenMP DLL conflicts between EasyOCR and PyTorch on AMD/CPU systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch, warnings
import easyocr
from sentence_transformers import SentenceTransformer
from PIL import Image

warnings.filterwarnings("ignore")

in_csv = 'bluesky_meme_dataset.csv'
img_dir = 'meme_images'
out_pkl = 'processed_features.pkl'

print("loading models...")
# Safely default to CPU if CUDA is unavailable or fails to initialize on AMD hardware
try:
    hw = 'cuda' if torch.cuda.is_available() else 'cpu'
except Exception:
    hw = 'cpu'
print("using:", hw)

ocr_reader = easyocr.Reader(['en'], gpu=(hw == 'cuda'))#optical character recognition
model = SentenceTransformer('clip-ViT-B-32', device=hw)#contrastive language pre training.

def extract_stuff():
    if not os.path.exists(in_csv):
        print("no csv found. run scraper first.")
        return

    data = pd.read_csv(in_csv)
    
    done_ids = set()
    if os.path.exists(out_pkl):
        old_pkl = pd.read_pickle(out_pkl)
        done_ids = set(old_pkl['post_id'].astype(str))
        
    # filter out already processed images
    to_process = data[~data['post_id'].astype(str).isin(done_ids)].copy()
    
    if to_process.empty:
        print("all caught up.")
        return

    print(f"processing {len(to_process)} images...")
    
    txt_list = []
    vecs = []
    kept_idx = []

    for idx, r in to_process.iterrows():
        path = os.path.join(img_dir, r['local_filename'])
        if not os.path.exists(path):
            continue
            
        try:
            # read text
            words = ocr_reader.readtext(path, detail=0)
            full_text = " ".join(words)
            
            # get embedding
            pic = Image.open(path).convert('RGB')
            embed = model.encode(pic)
            
            txt_list.append(full_text)
            vecs.append(embed)
            kept_idx.append(idx)
            
            if len(kept_idx) % 10 == 0:
                print(f"done {len(kept_idx)}")
        except Exception as err:
            print("issue with", r['local_filename'], err)

    res_df = to_process.loc[kept_idx].copy()
    res_df['extracted_text'] = txt_list
    res_df['image_embedding'] = vecs

    if os.path.exists(out_pkl):
        merged = pd.concat([old_pkl, res_df], ignore_index=True)
        merged.to_pickle(out_pkl)
    else:
        res_df.to_pickle(out_pkl)
        
    print("db updated")

if __name__ == "__main__":
    extract_stuff()