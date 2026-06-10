def analyze_sentiment(text: str):
    """Analyze the sentiment of the given text."""
    if "good" in text.lower():
        return "positive"
    elif "bad" in text.lower():
        return "negative"
    else:
        return "neutral"