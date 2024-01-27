from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForTokenClassification.from_pretrained("model")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "there have been many critiques of this view particularly political scientist elinor ostrom or economists amartya sen and esster boserup"
ner_results = nlp(example)
print(ner_results)
