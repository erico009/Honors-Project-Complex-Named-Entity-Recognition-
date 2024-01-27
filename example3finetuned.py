from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForTokenClassification.from_pretrained("model")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Lord Justice Scott Baker"
ner_results = nlp(example)
print(ner_results)
