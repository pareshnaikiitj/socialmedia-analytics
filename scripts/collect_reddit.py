import praw
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")
SUBREDDITS = ["marketing", "socialmedia"]
POST_LIMIT = 100
OUTPUT_PATH = "data/raw/reddit_raw.csv"

def collect_reddit_posts():
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )
    data = []
    for sub in SUBREDDITS:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.hot(limit=POST_LIMIT):
            data.append({
                "platform": "Reddit",
                "post_id": post.id,
                "author": str(post.author),
                "title": post.title,
                "text": post.selftext,
                "score": post.score,
                "created_utc": datetime.utcfromtimestamp(post.created_utc),
                "num_comments": post.num_comments
            })

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Collected {len(df)} Reddit posts → {OUTPUT_PATH}")

if __name__ == "__main__":
    collect_reddit_posts()
