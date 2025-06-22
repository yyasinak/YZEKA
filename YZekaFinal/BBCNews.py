import joblib
import gradio as gr
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

model = joblib.load("trained_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

def predict_category(text):
    if not text.strip():
        return "Lütfen bir haber içeriği girin."

    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    explanation = {
        "business": "Bu bir *iş* haberidir.",
        "entertainment": "Bu bir *eğlence* haberidir.",
        "politics": "Bu bir *siyaset* haberidir.",
        "sport": "Bu bir *spor* haberidir.",
        "tech": "Bu bir *teknoloji* haberidir."
    }

    return f"Tahmin edilen başlık: **{pred.upper()}**\n\n{explanation.get(pred, 'Kategori bulunamadı.')}"


interface = gr.Interface(
    fn=predict_category,
    inputs=gr.Textbox(lines=10, label="Haber İçeriğini Girin"),
    outputs=gr.Markdown(label="Tahmin Sonucu"),
    title="BBC Haber Başlık Tahmini",
    description="Aşağıya bir haber içeriği girin. Model, hangi kategoriye ait olduğunu tahmin etsin."
)

interface.launch()
