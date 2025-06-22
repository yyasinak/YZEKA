import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("bbc_news_text_complexity_summarization.csv")
X = df["text"]
y = df["labels"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=5, ngram_range=(1,2), max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n {model_name} - Doğruluk: {acc:.4f}")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=sorted(y.unique()))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
    plt.title(f"{model_name} - Karışıklık Matrisi")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.show()
    return acc

accuracies = {}

nb = MultinomialNB(alpha=0.5)
nb.fit(X_train_vec, y_train)
accuracies["Naive Bayes"] = evaluate_model(y_test, nb.predict(X_test_vec), "Naive Bayes")

dt = DecisionTreeClassifier(max_depth=30, min_samples_split=5, criterion='entropy')
dt.fit(X_train_vec, y_train)
accuracies["Karar Ağacı"] = evaluate_model(y_test, dt.predict(X_test_vec), "Karar Ağacı")


lr = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', class_weight='balanced', max_iter=1000)
lr.fit(X_train_vec, y_train)
accuracies["Lojistik Regresyon"] = evaluate_model(y_test, lr.predict(X_test_vec), "Lojistik Regresyon")

knn_cos = KNeighborsClassifier(n_neighbors=7, metric='cosine', weights='distance')
knn_cos.fit(X_train_vec, y_train)
accuracies["KNN - Cosine"] = evaluate_model(y_test, knn_cos.predict(X_test_vec), "KNN - Cosine")

acc_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Doğruluk"])

plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Doğruluk", data=acc_df)
plt.ylim(0.6, 1.0)
plt.title(" Modellerin Doğruluk Karşılaştırması")
plt.ylabel("Doğruluk Oranı")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
