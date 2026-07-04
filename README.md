# 🎬 Sentiment Analysis System

An end-to-end Natural Language Processing (NLP) project that analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative** using Machine Learning models and a Flask web application.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

This project performs sentiment analysis on movie reviews using Natural Language Processing (NLP) techniques and Machine Learning algorithms. The system includes:

- Text preprocessing using NLTK
- TF-IDF feature extraction
- Multiple model training and comparison
- Model evaluation using standard metrics
- Word Cloud visualization
- Interactive Flask web application
- REST API endpoint for predictions
- Vercel deployment support

---

## 🚀 Features

- ✅ Text Cleaning & Preprocessing
- ✅ Stopword Removal
- ✅ Lemmatization
- ✅ TF-IDF Vectorization
- ✅ Logistic Regression Model
- ✅ Multinomial Naive Bayes Model
- ✅ Linear SVM Model
- ✅ Model Comparison & Evaluation
- ✅ Confusion Matrix Visualization
- ✅ Positive & Negative Word Clouds
- ✅ Flask Web Interface
- ✅ REST API Support
- ✅ Vercel Deployment Ready
- ✅ Docker Support
- ✅ GitHub Actions CI/CD

---

## 🛠️ Tech Stack

| Category | Technologies |
|-----------|--------------|
| Programming Language | Python |
| Backend | Flask |
| Machine Learning | Scikit-Learn |
| NLP | NLTK |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, WordCloud |
| Frontend | HTML, CSS, Bootstrap |
| Deployment | Vercel |
| Version Control | Git & GitHub |

---

## 📂 Project Structure

```text
Sentiment-Analysis-System/
│
├── api/
│   └── index.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── logistic_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_comparison.ipynb
│
├── reports/
│   ├── confusion_matrix.png
│   ├── positive_wordcloud.png
│   ├── negative_wordcloud.png
│   └── model_performance.md
│
├── scripts/
│   ├── preprocess.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── generate_wordcloud.py
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── requirements.txt
├── vercel.json
├── Dockerfile
└── README.md
```

---

## 📊 Dataset

This project uses the **IMDb 50K Movie Reviews Dataset**.

🔗 **Download Link:**

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

After downloading, place the dataset in:

```text
data/raw/IMDB_Dataset.csv
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/anshuldeepbajpai-dhoni/Sentiment-Analysis-System.git
```

Move into the project directory:

```bash
cd Sentiment-Analysis-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧹 Text Preprocessing Pipeline

The preprocessing steps include:

- Convert text to lowercase
- Remove HTML tags
- Remove punctuation and numbers
- Remove stopwords
- Lemmatization using WordNet
- Clean and tokenize text

---

## 🤖 Machine Learning Models

The following models were trained and compared:

| Model | Purpose |
|---------|---------|
| Logistic Regression | Primary deployment model |
| Multinomial Naive Bayes | Baseline model |
| Linear SVM | Performance comparison |

---

## 📈 Model Evaluation

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Generated reports are stored inside:

```text
reports/
```

---

## ☁️ Generate Word Clouds

Generate positive and negative review word clouds:

```bash
python scripts/generate_wordcloud.py
```

---

## 🏋️ Train Models

Train all machine learning models:

```bash
python scripts/train_models.py
```

---

## 📊 Evaluate Models

Generate evaluation reports and confusion matrices:

```bash
python scripts/evaluate_models.py
```

---

## 🌐 Run Flask Application

Start the local server:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 🔌 API Usage

### Endpoint

```text
POST /api/predict
```

### Request

```json
{
    "review": "This movie was amazing!"
}
```

### Response

```json
{
    "review": "This movie was amazing!",
    "sentiment": "positive",
    "confidence": 95.2
}
```

---

## 🚀 Deployment

### Render Deployment 
Live link:-
https://sentiment-analysis-system-1-t4ew.onrender.com

Deploy steps:

1. Push code to GitHub
2. Import repository into Render
3. Deploy

---

## 🐳 Docker Support

Build Docker image:

```bash
docker build -t sentiment-analysis .
```

Run container:

```bash
docker run -p 5000:5000 sentiment-analysis
```

---

---

## 🔮 Future Enhancements

- BERT-based Sentiment Analysis
- Multi-language Support
- User Authentication
- Database Integration
- User Review History
- Advanced Analytics Dashboard
- Streamlit Version

---

## 👨‍💻 Author

**Anshul Deep Bajpai**

- GitHub: https://github.com/anshuldeepbajpai-dhoni
- LinkedIn: www.linkedin.com/in/anshuldeepbajpai

---

## ⭐ Support

If you found this project helpful, please consider giving it a **star ⭐** on GitHub.

---

## 📜 License

This project is licensed under the MIT License.
