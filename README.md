# Contradiction-Let-Me-Check

## Problem Statement

Description: For any two sentences, there are three ways they could be related: one could entail the other, one could contradict the other, or they could be unrelated. Natural Language Inferencing (NLI) is a popular NLP problem that involves determining how pairs of sentences (consisting of a premise and a hypothesis) arerelated. Your task is to create an NLI model that assigns labels of 0, 1, or 2 (corresponding to entailment, neutral, and contradiction) to pairs of premises and hypotheses.

## Idea

To make a contract analyzer that goes through large contract texts and helps the user understand whether the topic of concern is contradictory or not with the contract.IntroductionMost of us nowadays, sign legal contracts or documents without being able to read all of it and going into the tiniest details. But little do we know that these small details can be used for manipulating us and going through the legal procedure in our country is also a pain. Most of those affected by these mishaps are the uneducated and poor who can't read or write in their mother tongues too. What if there was an app that could assist regarding these matters? A friend or well-wisher you should consult before signing an important document? That is the purpose of our app.

## Description

Contract Contradiction Checker is an app, which helps you to do the following:

1. Scan through the longest documents within seconds and find the most appropriate result according to your search.

2. Mention whether the information you want to retrieve aligns with the contract text and give you appropriate answers as to whether, the text is a contradiction, a neutral statement, or an entailment.

3. This app was built to assist the poor and uneducated, who often fall into traps due to contract-based manipulation. Hence, we have tried to make it accessible to them using voice-based assistance and interactive elements.

## Usage

Contract Contradiction Checker (CCC) Gradio app can be accessed in the following ways:

1. Open the app available in the GitHub repository. Use the commands:- `pip install gradio sentence_transformers transformers torch stanza PyMuPDF selenium python-docx odfpy pypandoc sentencepiece ` to install dependencies, followed by `python main2.py`. Also, use `git checkout single-pretrained-model` to switch to the branch that has the same functionality but uses the single cross-encoder NLI DeBERTa model, thus saving resources and increasing accuracy.

2. In order to use the barebones NLI, head over to the "Main App" tab and choose any two languages. <br/><br/> ![Img1](./Imgs/NLI.png)<br><center>NLI App</center>

3. You can now add your sentences to check, in any of the languages that are required and the NLI will provide an output to you. <br/><br/> ![Img2](./Imgs/NLI2.png)<br><center>Output of NLI App</center>

4. Now, in order to use the main app, choose the language in which you will address your concern or problem. You will be greeted with a playable voice output and a translated welcome message in your language. <br/><br/> ![Img3](./Imgs/Welcome.png) <br><center>Language selection and Welcome</center>

5. Now enter the Topic of Concern and Detailed Description. You can do so by Transliteration, ASR and Translation, and headers of all of them can be translated into user languages as well for ease of access. <br/><br/> ![Entering inputs](./Imgs/Inputs.png) <br><center>Entering inputs</center>

6. After getting the text in your desired language, copy and paste your text using the copy option in the input dialog and paste by ctrl+v into the Topic of
   Concern box and Detail Problem Description Box. Hence, the Copy and Paste functionality is necessary. You can see the outputs of all the extra functionality below.

   ![Transliteration](./Imgs/Transliteration.png) <br><center>Transliteration</center><br/>
   
   ![Translation](./Imgs/Translation.png) <br/><center>Translation</center><br/>
   
   ![ASR](./Imgs/ASR.png) <br/><center>Automatic Speech Recognition</center><br/>

7. Now upload your contract text or image in the Upload your legal document box. You can provide text input also.
   ![Upload](./Imgs/Upload.png)<br/><center>Upload in the tab, along with button to enable text input</center><br/>

8. Select the language of the contract text/image and submit
   ![Text Input](./Imgs/Final%20Output%201.png) <br/><center>Text Input Along with lang, and output</center><br/>

9. You will receive the desired output, referring to whether you have a contradiction in any sentences in the related paragraphs of the legal document.
   ![Text Output](./Imgs/Final%20Output%202.png) <br/><center>The Text output below Tabular Output</center><br/>

10. Analyse the output and take further steps according to it.

11. Voice-based assistance according to user language will be provided at each step making the app more interactive and easier to use.
    ![Help](./Imgs/Help1.png) <br/><center>Voice+Translation based help for "Topic of Concern"</center><br/>

## Final Training and Eval loss for all languages

1.  Bn 0.717600 | 0.812157
2.  Hi 0.688700 | 0.785396
3.  Gu 0.745400 | 0.770488
4.  Pa 0.723800 | 0.764565
5.  Mr 0.677900 | 0.806058

## Resources

1. Web-parsed version of Sanskrit OCR using Selenium.

2. Semantic Search paraphrase-multilingual-mpnet-base-v2.3) Transliteration API (Source: Bhashini)

3. Translation API (Source: Bhashini)

4. ASR API (Source: Bhashini)

5. Fine-tuning NLI model Deberta Base V3 for Hindi, Bengali, Gujarati, Marathi, and Punjabi from IndicXNLI

6. Pretrained cross-encoder NLI Deberta V3 base on SNLI and MNLI.
7. Pretrained MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 to support the English language.

8. Document text reader using PyMuPDF

All resources are available on the GitHub repository.

# Project report File:-

View the project report file [here](https://docs.google.com/document/d/1IckT809b9js75s1wAiRs4aXBhnT6H45kTc2gKAhTuAE/)

# Model files:-

Get the model files [here](https://drive.google.com/drive/folders/153oDWFAo4CDtxjleLrd4HnqUcWRh8cl-?usp=drive_link).

Note: prefer to use single-pretrained-model branch because of it being more responsive, lightweight and accurate than this one.

# History and Other related references

- We also trained mDeberta Model on all the languages all at once, going with 3 epochs for each language. After 15 epochs, however, overfitting occurred and we had to abandon it. The model can be found [here](https://drive.google.com/drive/folders/1UrX23xcG1eBJNBD7WoexYqSLbLP8Fhe6?usp=drive_link).
- You can also check out our detailed documentation [here](https://docs.google.com/document/d/1NKI2h4Lz7j_lnOMguyerH1m0xiSmI6tOfOT7suvWww8/edit?usp=sharing)
