from textblob import TextBlob
import re
from numpy import arange
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.downloader.download('vader_lexicon')

def comment_cleaning(comment):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\ / \ / \S+)", " ", comment).split())

def get_comment_sentiment(comment):
    analysis = TextBlob(comment_cleaning(comment))
    polarity_score = 0
    polarity_score += analysis.polarity
    sid = SentimentIntensityAnalyzer()
    ss= sid.polarity_scores(comment)
    neg, neu, pos, compound = ss['neg'], ss['neu'], ss['pos'], ss['compound']
    polarity_score = compound - polarity_score + (neu/2)

    if polarity_score > 0:
        return 'positive'
    elif polarity_score == 0:
        return 'neutral'
    else:
        return 'negative'
