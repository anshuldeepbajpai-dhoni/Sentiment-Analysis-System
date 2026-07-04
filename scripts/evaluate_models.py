import joblib
import pandas as pd
import matplotlib.pyplot as plt

from preprocess import clean_text

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


df = pd.read_csv("data/raw/IMDB_Dataset.csv")

df["review"] = df["review"].apply(clean_text)

X = df["review"]
y = df["sentiment"]


vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

X = vectorizer.transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = joblib.load(
    "models/logistic_model.pkl"
)


predictions = model.predict(X_test)


report = classification_report(
    y_test,
    predictions
)

print(report)


with open(
    "reports/model_performance.md",
    "w"
) as file:

    file.write("# Model Performance\n\n")
    file.write("```text\n")
    file.write(report)
    file.write("\n```")


cm = confusion_matrix(
    y_test,
    predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.savefig(
    "reports/confusion_matrix.png"
)

plt.close()

print("Evaluation completed.")