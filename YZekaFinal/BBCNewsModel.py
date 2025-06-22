import pandas as pd
import re
import nltk
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("bbc_news_text_complexity_summarization.csv")

nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)               # Noktalama temizliği
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

df["text"] = df["text"].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["labels"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=5, ngram_range=(1, 2), max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    penalty="l2",
    solver="liblinear",
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

joblib.dump(model, "trained_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model başarıyla eğitildi ve kaydedildi!")
