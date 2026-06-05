import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load Dataset
df = pd.read_csv("twitter_training.csv", header=None)

# Rename Columns
df.columns = ["Tweet_ID", "Brand", "Sentiment", "Tweet"]

# Remove Missing Values
df.dropna(inplace=True)

# Dataset Info
print("Shape:", df.shape)
print(df.head())

# Sentiment Distribution

plt.figure(figsize=(8,5))
sns.countplot(x="Sentiment", data=df)

plt.title("Sentiment Distribution")
plt.savefig("sentiment_distribution.png")
plt.show()

# Pie Chart

sentiment_counts = df["Sentiment"].value_counts()

plt.figure(figsize=(7,7))
plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct="%1.1f%%"
)

plt.title("Sentiment Share")
plt.savefig("sentiment_pie_chart.png")
plt.show()

# Positive WordCloud

positive_text = " ".join(
    df[df["Sentiment"]=="Positive"]["Tweet"].astype(str)
)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(positive_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Positive Tweets WordCloud")
plt.savefig("positive_wordcloud.png")
plt.show()

# Negative WordCloud

negative_text = " ".join(
    df[df["Sentiment"]=="Negative"]["Tweet"].astype(str)
)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(negative_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Negative Tweets WordCloud")
plt.savefig("negative_wordcloud.png")
plt.show()

# Top 10 Brands

plt.figure(figsize=(12,6))

df["Brand"].value_counts().head(10).plot(kind="bar")

plt.title("Top 10 Brands in Dataset")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("top_brands.png")
plt.show()

print("Analysis Completed Successfully!")