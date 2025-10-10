import os
import argparse
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

import re
import math
import numpy as np
import pandas as pd
from datetime import datetime

# NLP / ML
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

# Optional embeddings
try:
    from sentence_transformers import SentenceTransformer
    SBT_AVAILABLE = True
except Exception:
    SBT_AVAILABLE = False

# Mongo
from pymongo import MongoClient, ASCENDING, TEXT, errors
from pymongo import UpdateOne

# -------------------------
# Utilities & preprocessing
# -------------------------
def basic_clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def parse_datetime(value):
    try:
        if pd.isna(value):
            return pd.NaT
        if isinstance(value, (int, float)):
            return pd.to_datetime(value, unit='s', utc=False)
        return pd.to_datetime(value)
    except Exception:
        return pd.NaT

# -------------------------
# Feature functions
# -------------------------
def compute_sentiment(texts):
    analyzer = SentimentIntensityAnalyzer()
    neg, neu, pos, comp = [], [], [], []
    for t in texts:
        if not isinstance(t, str) or t.strip() == "":
            scores = {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
        else:
            scores = analyzer.polarity_scores(t)
        neg.append(scores["neg"])
        neu.append(scores["neu"])
        pos.append(scores["pos"])
        comp.append(scores["compound"])
    return pd.DataFrame({
        "sent_neg": neg,
        "sent_neu": neu,
        "sent_pos": pos,
        "sent_compound": comp
    })

def compute_topics_gensim(texts, num_topics=5, no_below=5, no_above=0.5, keep_n=100000):
    tokenized = [simple_preprocess(t) for t in texts]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    if len(dictionary) == 0:
        logging.warning("Gensim dictionary empty => returning zeros for topic distributions.")
        return pd.DataFrame(np.zeros((len(texts), num_topics)),
                            columns=[f"topic_{i}" for i in range(num_topics)]), None

    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)
    doc_topics = []
    for bow in corpus:
        td = lda.get_document_topics(bow, minimum_probability=0.0)
        vec = [prob for (_tid, prob) in sorted(td, key=lambda x: x[0])]
        if len(vec) < num_topics:
            vec += [0.0] * (num_topics - len(vec))
        doc_topics.append(vec)
    cols = [f"topic_{i}" for i in range(num_topics)]
    return pd.DataFrame(doc_topics, columns=cols), lda

def compute_tfidf_lsa(texts, tfidf_max_features=5000, lsa_components=50):
    tfidf = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1,2), stop_words='english')
    X = tfidf.fit_transform(texts)
    if lsa_components is not None and lsa_components > 0:
        svd = TruncatedSVD(n_components=min(lsa_components, X.shape[1]-1 or 1), random_state=42)
        X_reduced = svd.fit_transform(X)
        cols = [f"tfidf_lsa_{i}" for i in range(X_reduced.shape[1])]
        return pd.DataFrame(X_reduced, columns=cols), tfidf, svd
    else:
        return X, tfidf, None

def compute_embeddings(texts, model_name="all-MiniLM-L6-v2", embed_dim=None):
    if not SBT_AVAILABLE:
        raise ImportError("sentence-transformers not installed. Install via 'pip install sentence-transformers'")
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    if embed_dim and embed_dim < emb.shape[1]:
        svd = TruncatedSVD(n_components=embed_dim, random_state=42)
        emb = svd.fit_transform(emb)
        return emb, model, svd
    return emb, model, None

def compute_engagement_features(df):
    s = df.get("score", pd.Series(0)).fillna(0).astype(float)
    c = df.get("num_comments", pd.Series(0)).fillna(0).astype(float)
    raw = np.log1p(s) * 0.6 + np.log1p(c) * 0.4
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(raw.values.reshape(-1,1)).flatten()
    return pd.DataFrame({
        "engagement_raw": raw,
        "engagement_norm": norm
    })

def compute_misc_text_features(texts):
    out = {"text_length": [], "word_count": [], "avg_word_len": [],
           "has_url": [], "uppercase_ratio": [], "punctuation_count": []}
    for t in texts:
        if not isinstance(t, str):
            t = ""
        out["text_length"].append(len(t))
        words = t.split()
        wc = len(words)
        out["word_count"].append(wc)
        out["avg_word_len"].append((sum(len(w) for w in words)/wc) if wc>0 else 0.0)
        out["has_url"].append(1 if re.search(r"http\S+", t) else 0)
        total_chars = len(t) if len(t)>0 else 1
        uppercase_chars = sum(1 for ch in t if ch.isupper())
        out["uppercase_ratio"].append(uppercase_chars / total_chars)
        out["punctuation_count"].append(len(re.findall(r"[^\w\s]", t)))
    return pd.DataFrame(out)

# -------------------------
# Mongo helpers
# -------------------------
def get_mongo_client():
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise ValueError("MONGO_URI not found in environment")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        logging.info("Connected to MongoDB successfully.")
    except Exception as e:
        logging.error(f"MongoDB connection failed: {e}")
        raise
    return client

def load_from_mongo(db_name="socialpulse", collection_name="reddit_posts", limit=None):
    client = get_mongo_client()
    coll = client[db_name][collection_name]
    cursor = coll.find().sort("created_utc", -1)
    if limit:
        cursor = cursor.limit(limit)
    df = pd.DataFrame(list(cursor))
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df

def upsert_features_to_mongo(db_name, collection_name, df):
    client = get_mongo_client()
    coll = client[db_name][collection_name]
    ops = [UpdateOne({"post_id": rec.get("post_id")}, {"$set": rec}, upsert=True)
           for rec in df.to_dict(orient="records")]
    if ops:
        try:
            result = coll.bulk_write(ops, ordered=False)
            logging.info("Mongo upsert completed: %s", result.bulk_api_result)
        except errors.BulkWriteError as bwe:
            logging.warning("BulkWriteError during upsert: %s", bwe.details)

def create_indexes(db_name="socialpulse", collection_name="reddit_posts"):
    client = get_mongo_client()
    coll = client[db_name][collection_name]
    try:
        coll.create_index([("clean_text", TEXT)], default_language="english")
    except Exception as e:
        logging.warning("Failed to create text index: %s", e)
    coll.create_index([("created_utc", ASCENDING)])
    coll.create_index([("post_id", ASCENDING)], unique=True)

# -------------------------
# Main pipeline
# -------------------------
def main(args):
    if args.source == "mongo":
        logging.info("Loading data from MongoDB %s/%s", args.mongo_db, args.mongo_collection)
        df = load_from_mongo(db_name=args.mongo_db, collection_name=args.mongo_collection, limit=args.limit)
    else:
        logging.info("Loading data from CSV: %s", args.csv_path)
        df = pd.read_csv(args.csv_path)

    if df.empty:
        logging.error("No data loaded. Exiting.")
        return

    if "clean_text" not in df.columns:
        df["clean_text"] = df.get("text", "").fillna("").astype(str).apply(basic_clean_text)
    else:
        df["clean_text"] = df["clean_text"].fillna("").astype(str)

    if "created_utc" in df.columns:
        df["created_utc_parsed"] = df["created_utc"].apply(parse_datetime)
    else:
        df["created_utc_parsed"] = pd.NaT

    texts = df["clean_text"].astype(str).tolist()

    logging.info("Computing VADER sentiment...")
    sent_df = compute_sentiment(texts)

    logging.info("Computing LDA topics (num_topics=%d)...", args.n_topics)
    topics_df, lda_model = compute_topics_gensim(texts, num_topics=args.n_topics, no_below=args.no_below, no_above=args.no_above)
    if lda_model and args.save_models:
        os.makedirs("models", exist_ok=True)
        lda_model.save(os.path.join("models", f"lda_{args.n_topics}.model"))
        logging.info("Saved LDA model to models/lda_%d.model", args.n_topics)

    logging.info("Computing TF-IDF + LSA (components=%d)...", args.lsa_components)
    tfidf_lsa_df, tfidf_vectorizer, lsa_model = compute_tfidf_lsa(texts, tfidf_max_features=args.tfidf_max_features, lsa_components=args.lsa_components)
    if args.save_models and tfidf_vectorizer:
        os.makedirs("models", exist_ok=True)
        import joblib
        joblib.dump(tfidf_vectorizer, os.path.join("models", "tfidf_vectorizer.joblib"))
        if lsa_model:
            joblib.dump(lsa_model, os.path.join("models", "lsa_svd.joblib"))

    embedding_df = None
    if args.embed:
        if not SBT_AVAILABLE:
            logging.error("sentence-transformers not installed. Install it or run without --embed.")
            return
        logging.info("Computing sentence-transformers embeddings with model '%s'...", args.embedding_model)
        emb_array, emb_model, emb_svd = compute_embeddings(texts, model_name=args.embedding_model, embed_dim=args.embed_dim)
        embedding_df = pd.DataFrame([list(vec) for vec in emb_array], columns=[f"embedding_{i}" for i in range(emb_array.shape[1])])

    logging.info("Computing engagement and misc features...")
    eng_df = compute_engagement_features(df)
    misc_df = compute_misc_text_features(texts)

    features = pd.concat([df.reset_index(drop=True),
                          sent_df.reset_index(drop=True),
                          topics_df.reset_index(drop=True),
                          tfidf_lsa_df.reset_index(drop=True),
                          eng_df.reset_index(drop=True),
                          misc_df.reset_index(drop=True)],
                         axis=1)

    if embedding_df is not None:
        features = pd.concat([features, embedding_df.reset_index(drop=True)], axis=1)

    cols = list(features.columns)
    if "post_id" in cols:
        cols = ["post_id"] + [c for c in cols if c!="post_id"]
        features = features[cols]

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    features.to_csv(args.output_csv, index=False)
    logging.info("Saved features CSV to %s (%d rows, %d cols)", args.output_csv, len(features), features.shape[1])

    if args.upsert_mongo:
        logging.info("Upserting features into MongoDB %s/%s", args.mongo_db, args.mongo_features_collection)
        upsert_features_to_mongo(args.mongo_db, args.mongo_features_collection, features)
        logging.info("Creating indexes on features collection...")
        create_indexes(db_name=args.mongo_db, collection_name=args.mongo_features_collection)
        logging.info("MongoDB upsert + indexing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering pipeline for social media posts")
    parser.add_argument("--source", choices=["csv", "mongo"], default="csv", help="Data source")
    parser.add_argument("--csv_path", default="data/cleaned/reddit_clean.csv", help="Path to cleaned CSV")
    parser.add_argument("--output_csv", default="data/features/reddit_features.csv", help="Output features CSV")
    parser.add_argument("--n_topics", type=int, default=6, help="Number of LDA topics")
    parser.add_argument("--tfidf_max_features", type=int, default=5000, help="TF-IDF max features")
    parser.add_argument("--lsa_components", type=int, default=50, help="Number of LSA components (TruncatedSVD)")
    parser.add_argument("--embed", action="store_true", help="Compute dense embeddings (requires sentence-transformers)")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--embed_dim", type=int, default=None, help="Reduce embedding to this dimension (optional)")
    parser.add_argument("--upsert_mongo", action="store_true", help="Upsert results to MongoDB features collection")
    parser.add_argument("--mongo_db", default="socialpulse", help="Mongo DB name")
    parser.add_argument("--mongo_collection", default="reddit_posts", help="Mongo source collection name")
    parser.add_argument("--mongo_features_collection", default="features", help="Mongo features collection name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to process")
    parser.add_argument("--no_below", type=int, default=5, help="Gensim filter_extremes no_below")
    parser.add_argument("--no_above", type=float, default=0.5, help="Gensim filter_extremes no_above")
    parser.add_argument("--save_models", action="store_true", help="Save models (lda, tfidf, lsa) to models/ directory")
    args = parser.parse_args()

    main(args)
