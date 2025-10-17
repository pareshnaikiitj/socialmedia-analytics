# Week 5: Topic Modeling Analysis
# --------------------------------
# Performs LDA-based topic modeling on cleaned + feature-enriched social media text data.
# Outputs topic keywords, coherence score, and top posts per topic.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import nltk
import os

# ===================== CONFIG ======================
DATA_PATH = "data/features/reddit_features.csv"
OUTPUT_TOPIC_FILE = "reports/week5_topic_summary.csv"
N_TOPICS = 8
N_TOP_WORDS = 12
os.makedirs("reports", exist_ok=True)
# ===================================================

# Download NLTK stopwords if not available
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords


def extract_topics(model, feature_names, n_top_words=12):
    # Extract top keywords for each topic.
    topic_list = []
    for i, topic in enumerate(model.components_):
        top_words = [feature_names[j] for j in topic.argsort()[:-n_top_words - 1:-1]]
        topic_list.append((i, top_words))
        print(f"Topic {i}: {', '.join(top_words)}")
    return topic_list


def compute_coherence(topics, texts):
    # Compute topic coherence (c_v) using Gensim.
    tokenized_texts = [t.split() for t in texts]
    dictionary = Dictionary(tokenized_texts)
    cm = CoherenceModel(
        topics=[words for _, words in topics],
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v',
        processes=1  # avoid multiprocessing on Windows
    )
    coherence_score = cm.get_coherence()
    return coherence_score


def main():
    # ============ LOAD DATA ============
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if 'text' not in df.columns:
        raise ValueError("The dataset must contain a 'text' column.")

    df['text'] = df['text'].astype(str)
    corpus = df['text'].tolist()
    print(f"[INFO] Loaded {len(df)} records for topic modeling.")

    # ============ PREPROCESS TEXT ============
    stop_words = set(stopwords.words('english'))
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=5,
        stop_words='english'
    )
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # ============ FIT LDA MODEL ============
    print("[INFO] Training LDA model...")
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=42,
        learning_method='batch'
    )
    lda.fit(X)
    print("[INFO] LDA model trained successfully.")

    # ============ EXTRACT TOPICS ============
    topics = extract_topics(lda, feature_names, N_TOP_WORDS)

    # ============ ASSIGN DOMINANT TOPIC ============
    doc_topic_matrix = lda.transform(X)
    df['dominant_topic'] = doc_topic_matrix.argmax(axis=1)
    df['topic_confidence'] = doc_topic_matrix.max(axis=1)

    # ============ COHERENCE SCORE ============
    print("[INFO] Calculating coherence score (this may take a minute)...")
    coherence = compute_coherence(topics, df['text'])
    print(f"\n[RESULT] Coherence Score (c_v): {coherence:.3f}")

    # ============ SAVE TOPIC SUMMARY ============
    topic_meta = []
    for t, words in topics:
        top_posts = df[df['dominant_topic'] == t] \
                        .sort_values('topic_confidence', ascending=False) \
                        .head(5)['text']
        topic_meta.append({
            'topic_id': t,
            'keywords': ", ".join(words),
            'example_posts': " || ".join(top_posts.tolist())
        })

    pd.DataFrame(topic_meta).to_csv(OUTPUT_TOPIC_FILE, index=False)
    print(f"\n[INFO] Saved topic summary → {OUTPUT_TOPIC_FILE}")

    # ============ OPTIONAL: SAVE LDA OUTPUT ============
    df.to_csv("data/features/social_features_with_topics.csv", index=False)
    print("[INFO] Saved dataset with topic assignments → data/features/social_features_with_topics.csv")


if __name__ == "__main__":
    main()
