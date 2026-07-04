import joblib
import pandas as pd

from preprocess import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score


df = pd.read_csv("data/raw/IMDB_Dataset.csv")

df["review"] = df["review"].apply(clean_text)

X = df["review"]
y = df["sentiment"]

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


models = {
    "logistic": LogisticRegression(),
    "naive_bayes": MultinomialNB(),
    "svm": LinearSVC()
}


for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"{name}: {accuracy:.4f}")

    joblib.dump(
        model,
        f"models/{name}_model.pkl"
    )


joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)