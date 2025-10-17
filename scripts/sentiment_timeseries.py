# Week 5: Time-Series Sentiment Analysis
# --------------------------------------
# Aggregates and visualizes sentiment trends over time.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

DATA_PATH = "data/features/reddit_features.csv"
REPORT_IMG = "reports/week5_sentiment_timeline.png"

os.makedirs("reports", exist_ok=True)

# =============== LOAD DATA ===============
df = pd.read_csv(DATA_PATH)
if 'sentiment' not in df.columns:
    raise ValueError("Sentiment column not found! Run feature_engineering.py first.")

if 'created_at' not in df.columns:
    raise ValueError("Timestamp column 'created_at' required for time series.")

df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df = df.dropna(subset=['created_at'])

# =============== AGGREGATE SENTIMENT ===============
df['date'] = df['created_at'].dt.date
daily = df.groupby('date').agg(mean_sentiment=('sentiment','mean'), count=('sentiment','size')).reset_index()
daily['rolling_mean'] = daily['mean_sentiment'].rolling(window=7, min_periods=1).mean()

# =============== PLOT SENTIMENT ===============
plt.figure(figsize=(14,6))
plt.plot(daily['date'], daily['mean_sentiment'], label='Daily Mean', color='skyblue', linewidth=2)
plt.plot(daily['date'], daily['rolling_mean'], label='7-Day Rolling Mean', color='orange', linewidth=3)
plt.scatter(daily['date'], daily['mean_sentiment'], s=(daily['count']/daily['count'].max())*100, alpha=0.5)
plt.title("Social Media Sentiment Over Time")
plt.xlabel("Date")
plt.ylabel("Average Sentiment (compound)")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_IMG, dpi=300)
plt.show()

print(f"[INFO] Sentiment timeline saved → {REPORT_IMG}")
