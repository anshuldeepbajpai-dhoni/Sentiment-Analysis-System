from flask import Flask, render_template, request, jsonify
import joblib

from scripts.preprocess import clean_text


app = Flask(__name__)


# Load model and vectorizer
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        review = request.form["review"]

        cleaned_review = clean_text(review)

        vector = vectorizer.transform([cleaned_review])

        prediction = model.predict(vector)[0]

        confidence = None

        # Only Logistic Regression supports probabilities
        if hasattr(model, "predict_proba"):

            confidence = round(
                max(model.predict_proba(vector)[0]) * 100,
                2
            )

        return render_template(
            "result.html",
            review=review,
            sentiment=prediction,
            confidence=confidence
        )

    except Exception as e:

        return render_template(
            "result.html",
            review="",
            sentiment=f"Error: {str(e)}",
            confidence=None
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():

    try:

        data = request.get_json()

        if not data or "review" not in data:

            return jsonify({
                "error": "Please provide a review."
            }), 400

        review = data["review"]

        cleaned_review = clean_text(review)

        vector = vectorizer.transform([cleaned_review])

        prediction = model.predict(vector)[0]

        confidence = None

        if hasattr(model, "predict_proba"):

            confidence = round(
                max(model.predict_proba(vector)[0]) * 100,
                2
            )

        return jsonify({
            "review": review,
            "sentiment": prediction,
            "confidence": confidence
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):

    return "<h1>404 - Page Not Found</h1>", 404


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )