import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from preprocess import clean_text


df = pd.read_csv("data/raw/IMDB_Dataset.csv")

df["review"] = df["review"].apply(clean_text)


positive_reviews = " ".join(
    df[df["sentiment"] == "positive"]["review"]
)

negative_reviews = " ".join(
    df[df["sentiment"] == "negative"]["review"]
)


positive_wc = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(positive_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(positive_wc)
plt.axis("off")
plt.title("Positive Reviews Word Cloud")
plt.savefig("reports/positive_wordcloud.png")
plt.close()


negative_wc = WordCloud(
    width=800,
    height=400,
    background_color="black"
).generate(negative_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(negative_wc)
plt.axis("off")
plt.title("Negative Reviews Word Cloud")
plt.savefig("reports/negative_wordcloud.png")
plt.close()

print("Word clouds generated successfully.")