import praw
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from pymongo import MongoClient
import schedule
import time
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# ====== NLTK setup ======
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# ====== MongoDB setup ======
MONGO_URI = os.getenv("MONGO_URI")  # make sure this is set in .env
if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in the .env file!")

client = MongoClient(MONGO_URI)
db = client["socialpulse"]
reddit_collection = db["reddit_posts"]

# ====== Reddit API setup ======
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# ====== Text cleaning function ======
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)   # remove Markdown links
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)   # remove punctuation
    text = text.lower()                           # lowercase
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

# ====== Reddit fetching function ======
def fetch_reddit_posts(subreddit_name="marketing", limit=50):
    posts = []
    subreddit = reddit.subreddit(subreddit_name)
    for submission in subreddit.new(limit=limit):
        posts.append({
            "platform": "Reddit",
            "post_id": submission.id,
            "author": str(submission.author),
            "title": submission.title,
            "text": submission.selftext,
            "score": submission.score,
            "created_utc": submission.created_utc,
            "num_comments": submission.num_comments,
            "clean_text": clean_text(submission.selftext)
        })
    if posts:
        reddit_collection.insert_many(posts)
    print(f"Inserted {len(posts)} Reddit posts from r/{subreddit_name}")

# ====== Pipeline runner ======
def run_pipeline():
    print("Running social media ingestion pipeline...")
    fetch_reddit_posts()
    print("Pipeline run completed.\n")

# ====== Scheduler ======
if __name__ == "__main__":
    # Run immediately
    run_pipeline()
    
    # Schedule every hour
    schedule.every(1).hours.do(run_pipeline)

    while True:
        schedule.run_pending()
        time.sleep(60)
