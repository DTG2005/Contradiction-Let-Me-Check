import gradio as gr
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from Bhashini_APIs import api_test_transliteration, api_test_translation, api_test_asr,api_test_tts
import torch
import base64
import time
# Importing functions from contractdocreader.py
from Main_Operations.docread import docread
from Main_Operations.semantic_search import semantic_search, tokenize_sentences


# Load models
semantic_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

nli_model_alt = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")

# Load NLI models for each language
nli_models = {
    "en": AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", num_labels=3),
    "hi": AutoModelForSequenceClassification.from_pretrained("./models/hi_model", num_labels=3),
    "mr": AutoModelForSequenceClassification.from_pretrained("./models/mr_model", num_labels=3),
    "gu": AutoModelForSequenceClassification.from_pretrained("./models/gu_model", num_labels=3),
    "pa": AutoModelForSequenceClassification.from_pretrained("./models/pa_model", num_labels=3),
    "bn": AutoModelForSequenceClassification.from_pretrained("./models/bn_model", num_labels=3),
}

def barebones_nli(premise, hypothesis, lang):
    model = nli_models[lang]
    model.eval()
    features = tokenizer([premise], [hypothesis],  padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**features).logits
        label_mapping = ['contradiction', 'entailment', 'neutral']
        labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
        return(labels)

def text_to_speech(text, lang, path):
    b64 = api_test_tts.tts(text, lang)
    audio_data = base64.b64decode(b64)
    with open(path, 'wb') as audio_file:
        audio_file.write(audio_data)
    return path, text

def NLI_language(text1, lang1, text2, lang2):
    if lang1 != lang2:
        text1 = api_test_translation.translate(text1, lang1, lang2)
    return barebones_nli(text1, text2, lang2)[0]

def save_base64_to_audio_file(base64_string, file_path):
    audio_data = base64.b64decode(base64_string)
    with open(file_path, 'wb') as audio_file:
        audio_file.write(audio_data)


# Initialize Bhashini API (mock for demonstration)
def bhashini_api_mock(text, source_lang, target_lang):
    # Mock function to represent translation, ASR, transliteration
    return api_test_translation.translate(text, source_lang=source_lang, tar_lang=target_lang)  # In actual implementation, connect to Bhashini API

# Welcome messages and audio files (mock paths)
welcome_messages = {
    "en": ("Welcome!", "welcome_audios/welcome.wav"),
    "hi": ("स्वागत है!", "welcome_audios/swagat.wav"),
    "mr": ("स्वागत आहे!", "welcome_audios/swagat_mr.wav"),
    "gu": ("સ્વાગત છે!", "welcome_audios/Swagat_chhe.wav"),
    "pa": ("ਸੁਆਗਤ ਹੈ!", "welcome_audios/welcome_pn.wav"),
    "bn": ("স্বাগতম!", "welcome_audios/welcome_bn.wav"),
}

# Function to handle language selection
def select_language(language):
    message, audio_path = welcome_messages[language]
    return message, audio_path

# Function to process the document
def process_document(language, topic, problem, document_stuff, document_file, document_lang):
    # Determine the document content based on the input type
    if document_stuff:
        document_text = document_stuff
    else:
        document_text = docread(document_file)
    if (document_lang!= language):
        ocr_text = bhashini_api_mock(document_text, document_lang, language)  # Translate to user's language
    else:
        ocr_text = document_text

    # Semantic search
    matched_paragraphs = semantic_search(topic, [ocr_text])
    results = []

    model = nli_models[language]  # Select the appropriate NLI model

    for para, similarity in matched_paragraphs:
        sentences = tokenize_sentences(para, language)
        for sentence in sentences:
            try:
                # Ensure the input is correctly formatted
                inputs = tokenizer(sentence, problem, return_tensors='pt', truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()
                label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
                predicted_label = label_map[predicted_class]
                print(f"Inputs to NLI model: {inputs}")  # Debugging line
                result = predicted_label
                results.append((sentence, result))
            except Exception as e:
                print(f"Error processing sentence: {sentence}, Error: {e}")

    
    translated_results = [(sentence, bhashini_api_mock(f"{result}", 'en', language)) for sentence, result in results]
    final_output_text = "\n".join([f"{sentence}\t{result}" for sentence, result in results])
    translated_output_text = "\n".join(f"{sentence}\t{result}" for sentence, result in translated_results)

    # Final output
    thank_you_message = bhashini_api_mock("Thank you!", "en", language)
    return results, translated_results, final_output_text, translated_output_text, thank_you_message

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Welcome! Please wait while we load...")
    
    with gr.Tabs():
        with gr.TabItem("Main App: NLI Model - Deberta"):
            language_1 = gr.Radio(["en", "hi", "mr", "gu", "pa", "bn"], label="Select premise language")
            txt = gr.Textbox(label="Enter your premise")
            language_2 = gr.Radio(["en", "hi", "mr", "gu", "pa", "bn"], label="Select hypothesis language")
            txt2 = gr.Textbox(label="Enter your hypothesis")
            txt3 = gr.Textbox(label="Output")
            gr.Button("Submit").click(NLI_language, inputs=[txt, language_1, txt2, language_2], outputs=txt3)

        with gr.TabItem("Step 1: Welcome"):
            language_req = gr.Radio(["en", "hi", "mr", "gu", "pa", "bn"], label="Select your language")
            welcome_output = gr.Textbox()
            welcome_audio = gr.Audio()
            gr.Button("Submit").click(select_language, inputs=language_req, outputs=[welcome_output, welcome_audio])

        with gr.TabItem("Step 2: Input Language and Details"):
            header_text_translated = gr.Textbox(label="Translated Header")
            header_audio = gr.Audio()
            gr.Button("Translate This").click(lambda lang: text_to_speech(bhashini_api_mock(user_topic.label, "en", lang), lang, "./audio_headers/header_1.wav"), inputs=language_req, outputs=[header_audio, header_text_translated])
            user_topic = gr.Textbox(label="Topic of Concern")
            user_problem = gr.Textbox(label="Detailed Problem Description")
            gr.Button("Translate This").click(lambda lang: text_to_speech(bhashini_api_mock(user_problem.label, "en", lang), lang, "./audio_headers/header_1.wav"), inputs=language_req, outputs=[header_audio, header_text_translated])
        
        with gr.TabItem("Step 3: Upload Document"):
            document = gr.File(label="Upload your legal document")
            input_textbox = gr.Textbox(label="Input", visible=False)
            document_lang = gr.Dropdown(["en", "hi", "mr", "gu", "pa", "bn"], label="Document Language")
            results_output = gr.Dataframe(headers=["Sentence", "NLI Result"])  # To display results of semantic search
            translated_output = gr.Dataframe(headers=["Translated Sentence", "Translated NLI Result"])
            original_text_output = gr.Textbox(label="Original Document Text", show_copy_button=True)
            translated_text_output = gr.Textbox(label="Translated Document Text", show_copy_button=True)
            # original_audio_output = gr.Audio(label="Original Audio Output")
            # translated_audio_output = gr.Audio(label="Translated Audio Output")
            def show_text():
                return {input_textbox: gr.Textbox(label="Input", visible=True)}
            thank_you_message = gr.Textbox(label="Thank You Message")
            # gr.Button("Help").click(process_document, inputs=[language_req, user_topic, user_problem, document, document_lang], outputs=[results_output, translated_output, original_text_output, translated_text_output, original_audio_output, translated_audio_output, thank_you_message])
            gr.Button("Enable Long Text Input").click(show_text, inputs=[], outputs=[input_textbox])
            gr.Button("Help").click(process_document, inputs=[language_req, user_topic, user_problem, input_textbox, document, document_lang], outputs=[results_output, translated_output, original_text_output, translated_text_output, thank_you_message])

        with gr.TabItem("Service 1: Transliteration (Only from english available so far)"):
            header_text = gr.Markdown("Transliteration (Only from English available so far)")
            header_text_translated = gr.Textbox(label="Translated Header")
            header_audio = gr.Audio()
            gr.Button("Translate Header").click(lambda lang: text_to_speech(bhashini_api_mock(header_text.value, "en", lang), lang, "./audio_headers/header_1.wav"), inputs=language_req, outputs=[header_audio, header_text_translated])
            srclanguage = gr.Radio(["en"], label="Select source language")
            gr.Button("Translate This").click(lambda lang: text_to_speech(bhashini_api_mock(srclanguage.label, "en", lang), lang, "./audio_headers/header_1.wav"), inputs=language_req, outputs=[header_audio, header_text_translated])
            language = gr.Radio(["en", "hi", "mr", "gu", "pa", "bn"], label="Select target language")
            gr.Button("Translate This").click(lambda lang: text_to_speech(bhashini_api_mock(language.label, "en", lang), lang, "./audio_headers/header_1.wav"), inputs=language_req, outputs=[header_audio, header_text_translated])
            textbox = gr.Textbox()
            transOutput = gr.Textbox(label="Output", show_copy_button=True)
            gr.Button("Submit").click(api_test_transliteration.transliteration, inputs=[textbox, srclanguage, language], outputs=transOutput)

        with gr.TabItem("Service 2: Translation (Only from english available so far)"):
            header_text = gr.Markdown("Translation (Only from English available so far)")
            header_text_translated = gr.Textbox(label="Translated Header")
            header_audio = gr.Audio()
            gr.Button("Translate Header").click(lambda lang: text_to_speech(bhashini_api_mock(header_text.value, "en", lang), lang, "./audio_headers/header_1.wav"), inputs=language_req, outputs=[header_audio, header_text_translated])
            srclanguage = gr.Radio(["en"], label="Select source language")
            gr.Button("Translate This").click(lambda lang: text_to_speech(bhashini_api_mock(srclanguage.label, "en", lang), lang, "./audio_headers/header_1.wav"), inputs=language_req, outputs=[header_audio, header_text_translated])
            language = gr.Radio(["en", "hi", "mr", "gu", "pa", "bn"], label="Select target language")
            gr.Button("Translate This").click(lambda lang: text_to_speech(bhashini_api_mock(language.label, "en", lang), lang, "./audio_headers/header_1.wav"), inputs=language_req, outputs=[header_audio, header_text_translated])
            textbox = gr.Textbox()
            transOutput = gr.Textbox(label="Output", show_copy_button=True)
            gr.Button("Submit").click(api_test_translation.translate, inputs=[textbox, srclanguage, language], outputs=transOutput)

        with gr.TabItem("Service 3: ASR"):
            header_text = gr.Markdown("Automatic Speech Recognition")
            header_text_translated = gr.Textbox(label="Translated Header")
            header_audio = gr.Audio()
            gr.Button("Translate Header").click(lambda lang: text_to_speech(bhashini_api_mock(header_text.value, "en", lang), lang, "./audio_headers/header_1.wav"), inputs=language_req, outputs=[header_audio, header_text_translated])
            language = gr.Radio(["en", "hi", "mr", "gu", "pa", "bn"], label="Select target language")
            gr.Button("Translate This").click(lambda lang: text_to_speech(bhashini_api_mock(language.label, "en", lang), lang, "./audio_headers/header_1.wav"), inputs=language_req, outputs=[header_audio, header_text_translated])
            audio = gr.Audio(type="filepath", label="Upload Audio")
            text = gr.Textbox(label="Output", show_copy_button=True)
            gr.Button("Submit").click(api_test_asr.asr, inputs=[audio, language], outputs=text)

demo.launch()