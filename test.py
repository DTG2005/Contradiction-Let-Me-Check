from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = r"./model"
tokenizer = AutoTokenizer.from_pretrained("microsoft/mDeberta-v3-base")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Input texts
premise = "હું એક છોકરો છું."
hypothesis = "હું છોકરો નથી."

# Tokenize and encode inputs
inputs = tokenizer.encode_plus(premise, hypothesis, return_tensors='pt')

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

# Output the result
for i in range(10):
    labels = ["Contradiction", "Neutral", "Entailment"]
    print(f"Predicted class: {labels[predicted_class]}")