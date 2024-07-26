from sentence_transformers import SentenceTransformer, util
from Main_Operations.docread import docread
from indicnlp.tokenize import sentence_tokenize

# Load Sentence Transformer model for semantic search
semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Function to get sentence embeddings
def get_embeddings(texts):
    return semantic_model.encode(texts, convert_to_tensor=True)

# Function to perform semantic search and filter relevant paragraphs
def semantic_search(query, documents, threshold=0.4):
    query_embedding = get_embeddings([query])
    doc_embeddings = get_embeddings(documents)
    similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    relevant_indices = [i for i in range(len(similarities)) if similarities[i] >= threshold]
    return [documents[i] for i in relevant_indices]

# Example usage
def main():
    user_text = "ब्रह्मपुत्र"
    contract_text = docread(r"C:\Users\divya\Desktop\Hentai\Hindi Project.docx")

    paragraphs = contract_text.split("\n")
    relevant_paragraphs = semantic_search(user_text, paragraphs)

    results = []
    for para in relevant_paragraphs:
        sentences = sentence_tokenize.sentence_split(para, lang='hi')  # Tokenizing the paragraph into sentences
        for sentence in sentences:
            similarity = util.pytorch_cos_sim(get_embeddings([user_text]), get_embeddings([sentence]))[0][0].item()
            if similarity > 0.4:  # Relevance threshold
                results.append((sentence, similarity))

    return results

print(main())