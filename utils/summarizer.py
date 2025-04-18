from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def generate_summary(text):
    return summarizer(text[:1024])[0]['summary_text']