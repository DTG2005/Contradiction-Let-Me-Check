import stanza
from Main_Operations.docread import docread

# Function to download the model and initialize the pipeline
def initialize_pipeline(language_code):
    stanza.download(language_code)
    return stanza.Pipeline(language_code)

# Function to tokenize text into sentences using Stanza
def tokenize_sentences(text, language_code):
    nlp = initialize_pipeline(language_code)
    doc = nlp(text)
    sentences = [sentence.text for sentence in doc.sentences]
    return sentences

# Example usage
text = docread(r"C:\Users\divya\Desktop\Hentai\Hindi Project.docx")
language_code = 'hi'  # Hindi, for example

sentences = tokenize_sentences(text, language_code)
print("Tokenized Sentences:", sentences)