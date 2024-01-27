from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Star Wars is a movie that takes place a long time ago in a galaxy far far away… It is about a war in the stars. It stars Mark Hamill."
ner_results = nlp(example)
print(ner_results)
