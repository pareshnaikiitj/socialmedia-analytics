import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

def clean_reddit(input_path='data/raw/reddit_raw.csv', output_path='data/cleaned/reddit_clean.csv'):
    df = pd.read_csv(input_path)
    df['title'] = df['title'].fillna('')
    df['clean_text'] = df['text'].apply(clean_text)
    df['clean_text'] = df['clean_text'].apply(remove_stopwords)
    df.to_csv(output_path, index=False)
    print(f"Cleaned Reddit data saved to {output_path}")

if __name__ == "__main__":
    clean_reddit()
