import gradio as gr
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from Bhashini_APIs import api_test_transliteration, api_test_translation, api_test_asr, api_test_tts
import nltk
import time
import base64
# Importing functions from contractdocreader.py
from Main_Operations.docread import docread
from Main_Operations.semantic_search import semantic_search, tokenize_sentences

def text_to_speech(text, lang, path):
    b64 = api_test_tts.tts(text, lang)
    audio_data = base64.b64decode(b64)
    with open(path, 'wb') as audio_file:
        audio_file.write(audio_data)
    return path, text

# Load models
semantic_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

nli_model_alt = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Load NLI models for each language
model = CrossEncoder('cross-encoder/nli-deberta-v3-base')



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
    # Pre-recorded audio files for headers
    "step_1": {
        "en": "audio/step_1_en.wav",
        "hi": "audio/step_1_hi.wav",
        "mr": "audio/step_1_mr.wav",
        "gu": "audio/step_1_gu.wav",
        "pa": "audio/step_1_pa.wav",
        "bn": "audio/step_1_bn.wav"
    },
    "service_1": {
        "en": "audio/service_1_en.wav",
        "hi": "audio/service_1_hi.wav",
        "mr": "audio/service_1_mr.wav",
        "gu": "audio/service_1_gu.wav",
        "pa": "audio/service_1_pa.wav",
        "bn": "audio/service_1_bn.wav"
    },
    # Add more entries for other headers
}

def save_base64_to_audio_file(base64_string, file_path):
    audio_data = base64.b64decode(base64_string)
    with open(file_path, 'wb') as audio_file:
        audio_file.write(audio_data)

# Function to handle language selection
def select_language(language):
    message, audio_path = welcome_messages[language]
    return message, audio_path

# # Function to get embeddings
# def get_embeddings(texts):
#     return semantic_model.encode(texts, convert_to_tensor=True)

# # Function to perform semantic search
# def semantic_search(query, documents, threshold=0.4):
#     query_embedding = get_embeddings([query])
#     doc_embeddings = get_embeddings(documents)
#     similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
#     relevant_indices = [i for i in range(len(similarities)) if similarities[i] >= threshold]
#     return [(documents[i], similarities[i].item()) for i in relevant_indices]

# Function to process the document
def process_document(language, topic, problem, document, document_lang):
    # Read and translate the document
    document_text = docread(document)
    prob_eng = ""
    if language != document_lang:
        topic = bhashini_api_mock(topic, language, document_lang)
        if language != "en":
            prob_eng = bhashini_api_mock(topic, language, "en")
        else:
            prob_eng = problem
        problem = bhashini_api_mock(topic, language, document_lang)

    # Semantic search
    matched_paragraphs = semantic_search(topic, [document_text])
    results = []
    for para, similarity in matched_paragraphs:
        sentences = tokenize_sentences(para, document_lang)
        for sentence in sentences:
            try:
                sentext = bhashini_api_mock(sentence, document_lang, "en") if document_lang != "en" else sentence
                scores = model.predict([(sentext, prob_eng)])
                label_mapping = ['contradiction', 'entailment', 'neutral']
                result = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
                results.append((sentence, result))
            except Exception as e:
                print(f"Error processing sentence: {sentence}, Error: {e}")

    # Translate the output back to the user's selected language
    translated_results = [(sentence, bhashini_api_mock(f"{result}", 'en', language)) for sentence, result in results]

    # Combine results into a single text
    final_output_text = "\n".join([f"{sentence}\t{result}" for sentence, result in results])
    translated_output_text = "\n".join(f"{sentence}\t{result}" for sentence, result in translated_results)

    # # Text-to-Speech for original and translated text
    # original_audio_base64 = api_test_tts.tts(final_output_text, source_lang=document_lang)
    # translated_audio_base64 = api_test_tts.tts(translated_output_text, source_lang=language)

    # original_audio_path = "original_output.wav"
    # translated_audio_path = "translated_output.wav"

    # save_base64_to_audio_file(original_audio_base64, original_audio_path)
    # save_base64_to_audio_file(translated_audio_base64, translated_audio_path)

    # Return the results, original document text, translated document text, and thank you message
    thank_you_message = "Thank you!"
    # return results, translated_results, final_output_text, translated_output_text, original_audio_path, translated_audio_path, thank_you_message
    return results, translated_results, final_output_text, translated_output_text, thank_you_message

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Welcome! Please wait while we load...")
    
    with gr.Tabs():
        with gr.TabItem("Step 1: Welcome"):
            language_req = gr.Radio(["en", "hi", "mr", "gu", "pa", "bn"], label="Select your language")
            welcome_output = gr.Textbox()
            welcome_audio = gr.Audio()
            gr.Button("Submit").click(select_language, inputs=language_req, outputs=[welcome_output, welcome_audio])

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

        with gr.TabItem("Step 2: Input Language and Details"):
            header_text_translated = gr.Textbox(label="Translated Header")
            header_audio = gr.Audio()
            gr.Button("Translate This").click(lambda lang: text_to_speech(bhashini_api_mock(user_topic.label, "en", lang), lang, "./audio_headers/header_1.wav"), inputs=language_req, outputs=[header_audio, header_text_translated])
            user_topic = gr.Textbox(label="Topic of Concern")
            user_problem = gr.Textbox(label="Detailed Problem Description")
            gr.Button("Translate This").click(lambda lang: text_to_speech(bhashini_api_mock(user_problem.label, "en", lang), lang, "./audio_headers/header_1.wav"), inputs=language_req, outputs=[header_audio, header_text_translated])
        
        with gr.TabItem("Step 3: Upload Document"):
            document = gr.File(label="Upload your legal document")
            document_lang = gr.Dropdown(["en", "hi", "mr", "gu", "pa", "bn"], label="Document Language")
            results_output = gr.Dataframe(headers=["Sentence", "NLI Result"])  # To display results of semantic search
            translated_output = gr.Dataframe(headers=["Translated Sentence", "Translated NLI Result"])
            original_text_output = gr.Textbox(label="Original Document Text", show_copy_button=True)
            translated_text_output = gr.Textbox(label="Translated Document Text", show_copy_button=True)
            # original_audio_output = gr.Audio(label="Original Audio Output")
            # translated_audio_output = gr.Audio(label="Translated Audio Output")
            thank_you_message = gr.Textbox(label="Thank You Message")
            # gr.Button("Help").click(process_document, inputs=[language_req, user_topic, user_problem, document, document_lang], outputs=[results_output, translated_output, original_text_output, translated_text_output, original_audio_output, translated_audio_output, thank_you_message])
            gr.Button("Help").click(process_document, inputs=[language_req, user_topic, user_problem, document, document_lang], outputs=[results_output, translated_output, original_text_output, translated_text_output, thank_you_message])

demo.launch()