from sentence_transformers import SentenceTransformer, util, CrossEncoder

# Load models
semantic_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Function to get embeddings
def get_embeddings(texts):
    return semantic_model.encode(texts, convert_to_tensor=True)

# Function to perform semantic search
def semantic_search(query, documents, threshold=0.4):
    query_embedding = get_embeddings([query])
    doc_embeddings = get_embeddings(documents)
    similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    relevant_indices = [i for i in range(len(similarities)) if similarities[i] >= threshold]
    return [(documents[i], similarities[i].item()) for i in relevant_indices]

print(semantic_search("pendulum", r"C:\Users\divya\Desktop\Hentai\Physics Activity 5.txt"))