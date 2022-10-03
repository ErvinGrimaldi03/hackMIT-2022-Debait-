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

# print(get_comment_sentiment("even though I think your opinion is stupid, fuck you idiot. But no for real, I'm sorry i'm just mad today. I  want to  tell you that i respect your opinion even though I don't agree with it"))
# print(get_comment_sentiment("fuck you, you're an idiot and you should shut the fuck up"))
# print(get_comment_sentiment("yeah no, I think that is stupid and Trump should shut up, but this is only my opinion you know??"))