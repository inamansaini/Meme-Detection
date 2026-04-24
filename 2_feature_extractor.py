import pandas as pd
import os
import easyocr
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import warnings

warnings.filterwarnings("ignore")

CSV_FILENAME = 'bluesky_meme_dataset.csv'
IMAGE_DIR = 'meme_images'
OUTPUT_FILENAME = 'processed_features.pkl'

print("Loading AI Models...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device.upper()}")

reader = easyocr.Reader(['en'], gpu=(device == 'cuda'))
clip_model = SentenceTransformer('clip-ViT-B-32', device=device)

def process():
    if not os.path.exists(CSV_FILENAME):
        print("Run 1_bluesky_scraper.py first!")
        return

    df = pd.read_csv(CSV_FILENAME)
    
    # If we already have a .pkl, only process NEW rows
    existing_processed_ids = set()
    if os.path.exists(OUTPUT_FILENAME):
        existing_df = pd.read_pickle(OUTPUT_FILENAME)
        existing_processed_ids = set(existing_df['post_id'].astype(str))
        
    new_rows = df[~df['post_id'].astype(str).isin(existing_processed_ids)].copy()
    
    if new_rows.empty:
        print("No new images to process.")
        return

    print(f"Extracting features for {len(new_rows)} new images...")
    
    extracted_texts, embeddings = [], []
    valid_indices = []

    for index, row in new_rows.iterrows():
        img_path = os.path.join(IMAGE_DIR, row['local_filename'])
        if not os.path.exists(img_path):
            continue
            
        try:
            # OCR
            ocr_text = " ".join(reader.readtext(img_path, detail=0))
            # CLIP
            img = Image.open(img_path).convert('RGB')
            emb = clip_model.encode(img)
            
            extracted_texts.append(ocr_text)
            embeddings.append(emb)
            valid_indices.append(index)
            
            if len(valid_indices) % 10 == 0:
                print(f"Processed {len(valid_indices)} images...")
        except Exception as e:
            print(f"Error on {row['local_filename']}: {e}")

    # Save results
    final_new_df = new_rows.loc[valid_indices].copy()
    final_new_df['extracted_text'] = extracted_texts
    final_new_df['image_embedding'] = embeddings

    if os.path.exists(OUTPUT_FILENAME):
        combined_df = pd.concat([existing_df, final_new_df], ignore_index=True)
        combined_df.to_pickle(OUTPUT_FILENAME)
    else:
        final_new_df.to_pickle(OUTPUT_FILENAME)
        
    print(f"SUCCESS! Database updated.")

if __name__ == "__main__":
    process()