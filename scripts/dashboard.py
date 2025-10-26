# Week 6: Dashboard and Final Report Generation
# ---------------------------------------------
# Creates visual dashboards for sentiment trends, topic distribution,
# and hashtag analysis using Matplotlib, Seaborn, and Plotly.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from wordcloud import WordCloud

# ========== CONFIG ==========
FEATURE_FILE = "data/features/social_features_with_topics.csv"
REPORT_DIR = "reports/week6_dashboard"
os.makedirs(REPORT_DIR, exist_ok=True)
sns.set(style="whitegrid")

# ========== LOAD DATA ==========
df = pd.read_csv(FEATURE_FILE)
# ---------- Handle timestamp column ----------
if 'created_at' in df.columns:
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
elif 'created_utc_parsed' in df.columns:
    df['created_at'] = pd.to_datetime(df['created_utc_parsed'], errors='coerce')
elif 'created_utc' in df.columns:
    df['created_at'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
elif 'date' in df.columns:
    df['created_at'] = pd.to_datetime(df['date'], errors='coerce')
else:
    raise ValueError("No valid timestamp column found.")

# ---------- Handle sentiment column ----------
if 'sentiment' in df.columns:
    df['sentiment'] = df['sentiment']
elif 'sent_compound' in df.columns:
    df['sentiment'] = df['sent_compound']
else:
    raise ValueError("No sentiment column found (expected 'sentiment' or 'sent_compound').")

df = df.dropna(subset=['created_at'])
df['date'] = df['created_at'].dt.date
df['hour'] = df['created_at'].dt.hour

print(f"[INFO] Loaded {len(df)} records for dashboard generation.")

# ========== SENTIMENT HEATMAP ==========
df['date'] = df['created_at'].dt.date
df['hour'] = df['created_at'].dt.hour
heatmap_data = df.groupby(['date', 'hour'])['sentiment'].mean().unstack()

plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_data, cmap="coolwarm", center=0)
plt.title("Sentiment Heatmap Over Time (Date vs Hour)")
plt.xlabel("Hour of Day")
plt.ylabel("Date")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/sentiment_heatmap.png", dpi=300)
plt.close()
print("[INFO] Saved sentiment heatmap.")

# ========== TOPIC DISTRIBUTION ==========
if 'dominant_topic' in df.columns:
    topic_counts = df['dominant_topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic ID', 'Post Count']

    plt.figure(figsize=(10, 5))
    sns.barplot(data=topic_counts, x='Topic ID', y='Post Count', palette="Set2")
    plt.title("Distribution of Topics Across Posts")
    plt.xlabel("Topic ID")
    plt.ylabel("Number of Posts")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/topic_distribution.png", dpi=300)
    plt.close()
    print("[INFO] Saved topic distribution plot.")

# ========== HASHTAG ANALYSIS ==========
def extract_hashtags(text):
    import re
    return [tag.strip("#") for tag in re.findall(r"#\w+", str(text).lower())]

df['hashtags'] = df['text'].apply(extract_hashtags)
all_hashtags = [h for tags in df['hashtags'] for h in tags]
hashtag_series = pd.Series(all_hashtags).value_counts().head(15)

plt.figure(figsize=(10, 5))
sns.barplot(x=hashtag_series.values, y=hashtag_series.index, palette="crest")
plt.title("Top 15 Hashtags")
plt.xlabel("Usage Count")
plt.ylabel("Hashtag")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/top_hashtags.png", dpi=300)
plt.close()
print("[INFO] Saved top hashtags plot.")

# ========== WORDCLOUD (Top Words by Sentiment) ==========
positive_text = " ".join(df[df['sentiment'] > 0.2]['text'].astype(str))
negative_text = " ".join(df[df['sentiment'] < -0.2]['text'].astype(str))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
wc_pos = WordCloud(width=800, height=400, background_color="white", colormap="Greens").generate(positive_text)
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis("off")
plt.title("Positive Sentiment WordCloud")

plt.subplot(1, 2, 2)
wc_neg = WordCloud(width=800, height=400, background_color="white", colormap="Reds").generate(negative_text)
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis("off")
plt.title("Negative Sentiment WordCloud")

plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/sentiment_wordclouds.png", dpi=300)
plt.close()
print("[INFO] Saved sentiment wordclouds.")

# ========== INTERACTIVE DASHBOARD (Plotly) ==========
# Sentiment trend
sent_trend = df.groupby('date')['sentiment'].mean().reset_index()
fig1 = px.line(sent_trend, x='date', y='sentiment',
               title="Daily Average Sentiment Trend",
               line_shape='spline', markers=True)
fig1.write_html(f"{REPORT_DIR}/interactive_sentiment_trend.html")

# Topic popularity (if exists)
if 'dominant_topic' in df.columns:
    topic_trend = df.groupby(['date', 'dominant_topic']).size().reset_index(name='count')
    fig2 = px.area(topic_trend, x='date', y='count', color='dominant_topic',
                   title="Topic Popularity Over Time")
    fig2.write_html(f"{REPORT_DIR}/interactive_topic_trend.html")

# Hashtag usage interactive chart
if len(all_hashtags) > 0:
    hashtag_df = pd.Series(all_hashtags).value_counts().reset_index()
    hashtag_df.columns = ['hashtag', 'count']
    fig3 = px.bar(hashtag_df.head(20), x='hashtag', y='count',
                  title="Top Hashtags (Interactive)")
    fig3.write_html(f"{REPORT_DIR}/interactive_hashtags.html")

print(f"[INFO] Interactive dashboards saved to {REPORT_DIR}/")

# ========== FINAL SUMMARY ==========
summary = {
    "Total Posts": len(df),
    "Date Range": f"{df['date'].min()} → {df['date'].max()}",
    "Avg Sentiment": round(df['sentiment'].mean(), 3),
    "Most Common Topic": df['dominant_topic'].mode()[0] if 'dominant_topic' in df.columns else "N/A",
    "Top Hashtag": hashtag_series.index[0] if not hashtag_series.empty else "N/A"
}

summary_df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
summary_df.to_csv(f"{REPORT_DIR}/final_summary.csv", index=False)
print("[INFO] Final summary saved.")
print(summary_df)
