from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the fine-tuned model and tokenizer
model_path = r'cross-encoder/nli-deberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-base")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create a pipeline for text classification
nli_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

def classify_nli(premise, hypothesis):
    input_text = f"{premise} [SEP] {hypothesis}"
    result = nli_pipeline(input_text)
    return result[0]['label']

def generate_candidate_hypotheses(premise):
    # This is a mock function. In practice, use a language generation model like GPT-3.
    return [
        "તાપમાન ક્યારેય શૂન્યથી નીચે જતું નથી.",
        "શિયાળામાં તાપમાન ઊંચું રહે છે.",
        "ઉનાળામાં ઠંડી હોય છે."
    ]

def generate_contradictions(premise):
    candidates = generate_candidate_hypotheses(premise)
    contradictions = []

    for hypothesis in candidates:
        label = classify_nli(premise, hypothesis)
        if label == 'contradiction':
            contradictions.append(hypothesis)

    return contradictions

# Example usage
premise = "શિયાળામાં તાપમાન ઘણીવાર શૂન્યથી નીચે જાય છે."
contradictions = generate_contradictions(premise)
print(contradictions)  # This will print out the hypotheses classified as contradictions