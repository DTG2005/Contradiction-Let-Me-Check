from sentence_transformers import SentenceTransformer, util
import stanza

# Function to download the model and initialize the pipeline
def initialize_pipeline(language_code):
    if language_code=="pa" or language_code == "gu":
        language_code = 'en'
    stanza.download(language_code)
    return stanza.Pipeline(language_code)

# Function to tokenize text into sentences using Stanza
def tokenize_sentences(text : str, language_code):
    nlp = initialize_pipeline(language_code)
    doc = nlp(text)
    sentences = [sentence.text for sentence in doc.sentences]
    if language_code == "gu" or language_code == "pa":
        lisText = text.split('.')
        return lisText
    return sentences

# Load model
semantic_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Function to get embeddings
def get_embeddings(texts):
    return semantic_model.encode(texts, convert_to_tensor=True)

# Function to perform semantic search
def semantic_search(query, sentences, threshold=0.1):  # Lower threshold for testing
    query_embedding = get_embeddings([query])
    sentence_embeddings = get_embeddings(sentences)
    similarities = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]

    # Debug prints
    print("Query Embedding Shape:", query_embedding.shape)
    print("Sentence Embeddings Shape:", sentence_embeddings.shape)
    print("Similarities:", similarities)

    relevant_indices = [i for i in range(len(similarities)) if similarities[i] >= threshold]
    return [(sentences[i], similarities[i].item()) for i in relevant_indices]