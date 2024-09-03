from transformers import pipeline
import sys

classifier = pipeline('sentiment-analysis', device=0)

text = sys.argv[1]

result = classifier(text)[0]

print(f"The text \"{text}\" was classified as {result['label']} with a score of {round(result['score'], 4) * 100}%")
