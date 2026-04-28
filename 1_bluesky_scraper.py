from atproto import Client
import pandas as pd
import requests
import os
import time
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
BLUESKY_HANDLE = 'namansaini720.bsky.social' # e.g., naman.bsky.social
BLUESKY_APP_PASSWORD = 'worldisfreee4u'

# ==========================================
# 1. UPDATED CONFIGURATION (HIGHER VOLUME)
# ==========================================
# Add specific sub-niches to get more variety
SEARCH_TERMS = [
    'meme', 'dank meme', 'programming meme', 
    'gaming meme', 'relatable meme', 'shitpost', 
    'cat meme', 'crypto meme'
]
POSTS_PER_TERM = 1000  # Increased from 50 to 100
IMAGE_DIR = 'meme_images'
CSV_FILENAME = 'bluesky_meme_dataset.csv'

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# ==========================================
# HELPER: DOWNLOAD IMAGE
# ==========================================
def download_image(url, filename):
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(os.path.join(IMAGE_DIR, filename), 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        return False
    except Exception as e:
        return False

# ==========================================
# MAIN SCRAPER LOGIC
# ==========================================
def scrape_bluesky():
    print("Connecting to Bluesky...")
    client = Client()
    try:
        client.login(BLUESKY_HANDLE, BLUESKY_APP_PASSWORD)
    except Exception as e:
        print(f"Login failed: {e}. Check your handle and app password.")
        return

    # Load existing data to prevent duplicates
    existing_ids = set()
    if os.path.exists(CSV_FILENAME):
        existing_df = pd.read_csv(CSV_FILENAME)
        existing_ids = set(existing_df['post_id'].astype(str))
    
    new_data = []
    
    for term in SEARCH_TERMS:
        print(f"\nSearching Bluesky for: '{term}'")
        try:
            # Search for posts containing the term
            response = client.app.bsky.feed.search_posts({'q': term, 'limit': POSTS_PER_TERM})
            
            for post in response.posts:
                post_id = post.uri.split('/')[-1]
                
                # Skip if we already have it
                if post_id in existing_ids:
                    continue
                
                # Check if the post has an image attached
                if hasattr(post.embed, 'images') and post.embed.images:
                    image_data = post.embed.images[0]
                    image_url = image_data.fullsize
                    
                    file_extension = 'jpg' # Bluesky generally serves JPEGs for fullsize
                    filename = f"{post_id}.{file_extension}"
                    
                    if download_image(image_url, filename):
                        new_data.append({
                            'post_id': post_id,
                            'author_handle': post.author.handle,
                            'text': post.record.text,
                            'likes': post.like_count,
                            'reposts': post.repost_count,
                            'timestamp': post.record.created_at, # REAL TIMESTAMP!
                            'local_filename': filename
                        })
                        existing_ids.add(post_id)
                        print(f"Downloaded: {post.record.text[:30]}...")
            
            time.sleep(1) # Be polite to the API
            
        except Exception as e:
            print(f"Error searching for {term}: {e}")

    # Save to CSV
    if new_data:
        df = pd.DataFrame(new_data)
        if os.path.exists(CSV_FILENAME):
            df.to_csv(CSV_FILENAME, mode='a', header=False, index=False)
        else:
            df.to_csv(CSV_FILENAME, index=False)
        print(f"\nSaved {len(new_data)} new posts to {CSV_FILENAME}.")
    else:
        print("\nNo new image posts found.")

if __name__ == "__main__":
    scrape_bluesky()