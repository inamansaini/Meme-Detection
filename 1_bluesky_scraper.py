import os
from atproto import Client # use to login into bluesky environment using credentials
import pandas as pd
import requests, time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
user_handle = os.getenv('BSKY_HANDLE')
app_pwd = os.getenv('BSKY_PASSWORD')

tags_to_search = ['programming meme', 'gaming meme', 'hindi good meme', 'hindi cute memes', 'AI memes', 'trump memes']
limit_per_tag = 50
img_folder = 'meme_images'
csv_db = 'bluesky_meme_dataset.csv'

# A set of common adult/NSFW labels used by Bluesky moderation
nsfw_labels = {'porn', 'sexual', 'nudity', 'graphic-media', 'nsfw'}

if not os.path.exists(img_folder):
    os.makedirs(img_folder)

def fetch_img(url, name):
    try:
        res = requests.get(url, stream=True, timeout=10)
        if res.status_code == 200:
            with open(os.path.join(img_folder, name), 'wb') as f:
                for c in res.iter_content(1024):
                    f.write(c)
            return True
        return False
    except:
        return False

def run_scraper():
    print("logging in...")
    client = Client()
    try:
        client.login(user_handle, app_pwd)
    except Exception as err:
        print("login err:", err)
        return

    seen = set()
    if os.path.exists(csv_db):
        old_data = pd.read_csv(csv_db)
        seen = set(old_data['post_id'].astype(str))
    
    collected = []
    
    for tag in tags_to_search:
        print(f"\nlooking up: '{tag}'")
        try:
            feed = client.app.bsky.feed.search_posts({'q': tag, 'limit': limit_per_tag})
            
            for p in feed.posts:
                pid = p.uri.split('/')[-1]
                
                # skip already seen posts
                if pid in seen:
                    continue
                
                # skip posts or authors marked with adult labels
                is_nsfw = False
                if hasattr(p, 'labels') and p.labels:
                    if any(lbl.val in nsfw_labels for lbl in p.labels):
                        is_nsfw = True
                if hasattr(p.author, 'labels') and p.author.labels:
                    if any(lbl.val in nsfw_labels for lbl in p.author.labels):
                        is_nsfw = True
                        
                if is_nsfw:
                    continue
                
                # check if it actually has an image
                if hasattr(p.embed, 'images') and p.embed.images:
                    img_info = p.embed.images[0]
                    dl_link = img_info.fullsize
                    fname = f"{pid}.jpg"
                    
                    if fetch_img(dl_link, fname):
                        collected.append({
                            'post_id': pid,
                            'author_handle': p.author.handle,
                            'text': p.record.text,
                            'likes': p.like_count,
                            'reposts': p.repost_count,
                            'timestamp': p.record.created_at, 
                            'local_filename': fname
                        })
                        seen.add(pid)
                        print(f"got: {p.record.text[:30]}...")
            time.sleep(1)
        except Exception as ex:
            print(f"failed on {tag}:", ex)

    if len(collected) > 0:
        new_df = pd.DataFrame(collected)
        if os.path.exists(csv_db):
            new_df.to_csv(csv_db, mode='a', header=False, index=False)
        else:
            new_df.to_csv(csv_db, index=False)
        print(f"\nsaved {len(collected)} items.")
    else:
        print("\nnothing new found")

if __name__ == "__main__":
    run_scraper()