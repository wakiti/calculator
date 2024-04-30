from textblob import TextBlob

# Sample text (replace this with your actual review data)
sample_review = "This laptop is amazing! It's fast and powerful."

# Perform sentiment analysis using TextBlob
analysis = TextBlob(sample_review)

# Get sentiment polarity (-1 to 1, where <0 is negative, 0 is neutral, >0 is positive)
sentiment_score = analysis.sentiment.polarity

if sentiment_score > 0:
    print("Positive sentiment")
elif sentiment_score == 0:
    print("Neutral sentiment")
else:
    print("Negative sentiment")
