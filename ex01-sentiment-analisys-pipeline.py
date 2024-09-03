from transformers import pipeline

# need to add device=0 to have it run on M1
classifier = pipeline('sentiment-analysis', device=0)

text = "I love coding in python!"

result = classifier(text)[0]

print(f"The text \"{text}\" was classified as {result['label']} with a score of {round(result['score'], 4) * 100}%")
