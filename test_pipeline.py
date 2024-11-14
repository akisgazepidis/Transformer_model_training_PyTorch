from transformers import pipeline

classifier = pipeline("sentiment-analysis", device=0)
print(classifier("we love you"))
