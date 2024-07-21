import gradio as gr
from sentence_transformers import SentenceTransformer, util
from Main_Operations.semantic_search import semantic_search
from transformers import pipeline
import nltk

nltk.download('punkt')

model = "./model/" #The model file that will be locally available after training
model_alternate = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" #The model that will be loaded in case of a backup

target_accuracy= 0.4

# Load models
semantic_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
try:
    nli_model = pipeline("text-classification", model=model)  # Update with your model path
except:
    nli_model = pipeline("text-classification", model=model_alternate) 

# Initialize Bhashini API (mock for demonstration)
def bhashini_api_mock(text, source_lang, target_lang):
    # Mock function to represent translation, ASR, transliteration
    return text  # In actual implementation, connect to Bhashini API

# Welcome messages and audio files (mock paths)
welcome_messages = {
    "en": ("Welcome!", "welcome_audios/welcome.wav"),
    "hi": ("स्वागत है!", "welcome_audios/swagat.wav"),
    "mr": ("स्वागत आहे!", "welcome_audios/swagat_mr.wav"),
    "gu": ("સ્વાગત છે!", "welcome_audios/Swagat_chhe.wav"),
    "pa": ("ਸੁਆਗਤ ਹੈ!", "welcome_audios/welcome_pn.wav"),
    "bn": ("স্বাগতম!", "welcome_audios/welcome_bn.wav"),
}

# Map full language names to codes
language_map = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Bengali": "bn"
}

# Function to handle language selection
def select_language(language):
    language_code = language_map[language]
    message, audio_path = welcome_messages[language_code]
    return message, audio_path

def final_process(text : str, hypothesis: str):
    # Perform semantic search
    paragraphs = text.split('\n')
    matched_paragraphs = semantic_search(hypothesis, paragraphs)
    
    # NLI classification
    results = []
    for para, similarity in matched_paragraphs:
        sentences = nltk.sent_tokenize(para)
        for sentence in sentences:
            result = nli_model({"premise": sentence, "hypothesis": hypothesis})
            results.append((sentence, result))

    return results

# Function to process the document
def process_document(language, topic, problem, document, document_lang):
    # OCR and translation (mock for demonstration)
    ocr_text = bhashini_api_mock(document, document_lang, language)  # Perform OCR and translate to user's language

    # Perform semantic search
    paragraphs = ocr_text.split('\n')
    matched_paragraphs = semantic_search(topic, paragraphs)
    
    # NLI classification
    results = []
    for para, similarity in matched_paragraphs:
        sentences = nltk.sent_tokenize(para)
        for sentence in sentences:
            result = nli_model({"premise": sentence, "hypothesis": problem})
            results.append((sentence, result))

    # Thank you message
    thank_you_message = bhashini_api_mock("Thank you!", "en", language)

    return results, thank_you_message

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Welcome! Please wait while we load...")

    with gr.Tabs():
        with gr.TabItem("Step 1: Welcome"):
            language = gr.Radio(["English", "Hindi", "Marathi", "Gujarati", "Punjabi", "Bengali"], label="Select your language")
            welcome_output = gr.Textbox()
            welcome_audio = gr.Audio()
            gr.Button("Submit").click(select_language, inputs=language, outputs=[welcome_output, welcome_audio])

        with gr.TabItem("Step 2: Input Language and Details"):
            user_language = gr.Radio(["en", "hi", "mr", "gu", "pa", "bn"], label="Preferred Language")
            user_topic = gr.Textbox(label="Topic of Concern")
            user_problem = gr.Textbox(label="Detailed Problem Description")
        
        with gr.TabItem("Step 3: Upload Document"):
            document = gr.File(label="Upload your legal document")
            document_lang = gr.Dropdown(["English", "Hindi", "Marathi", "Gujarati", "Punjabi", "Bengali"], label="Document Language")
            results_output = gr.Dataframe(headers=["Sentence", "NLI Result"])  # To display results of semantic search
            thank_you_message = gr.Textbox(label="Thank You Message")
            gr.Button("Submit").click(process_document, inputs=[user_language, user_topic, user_problem, document, document_lang], outputs=[results_output, thank_you_message])

        with gr.TabItem("Test 1: TextBoxes for NLI activity:1"):
            paragraphs_1 = gr.Textbox(label="Paragraph goes here.", lines=3)
            hypothesis_1 = gr.Textbox(label="Hypothesis goes here.")
            Output = gr.Textbox(label="Output")
            gr.Button("Submit").click(final_process, inputs=[paragraphs_1, hypothesis_1], outputs=[Output])

demo.launch()