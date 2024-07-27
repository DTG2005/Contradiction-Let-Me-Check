# Contradiction-Let-Me-Check


## Problem Statement

Description: For any two sentences, there are three ways they could be related: one could entail the other, one could contradict the other, or they could be unrelated. Natural Language Inferencing (NLI) is a popular NLP problem that involves determining how pairs of sentences (consisting of a premise and a hypothesis) arerelated. Your task is to create an NLI model that assigns labels of 0, 1, or 2 (corresponding to entailment, neutral, and contradiction) to pairs of premises and hypotheses.

## Idea

To make a contract analyzer that goes through large contract texts and helps the user understand whether the topic of concern is contradictory or not with the contract.IntroductionMost of us nowadays, sign legal contracts or documents without being able to read all of it and going into the tiniest details. But little do we know that these small details can be used for manipulating us and going through the legal procedure in our country is also a pain. Most of those affected by these mishaps are the uneducated and poor who can't read or write in their mother tongues too. What if there was an app that could assist regarding these matters? A friend or well-wisher you should consult before signing an important document? That is the purpose of our app.

## Description

Contract Contradiction Checker is an app, which helps you to do the following:

1) Scan through the longest documents within seconds and find the most appropriate result according to your search.

2) Mention whether the information you want to retrieve aligns with the contract text and give you appropriate answers as to whether, the text is a contradiction, a neutral statement, or an entailment.

3) This app was built to assist the poor and uneducated, who often fall into traps due to contract-based manipulation. Hence, we have tried to make it accessible to them using voice-based assistance and interactive elements.

## Usage

 Contract Contradiction Checker (CCC) Gradio app can be accessed in the following ways:

1) Open the app available in the GitHub repository.

2) Now, choose the language in which you will address your concern or problem.

3) Now enter the Topic of Concern and Detailed Description. You can do so by Transliteration, ASR and Translation.

4) After getting the text in your desired language, copy and paste your text using the copy option in the input dialog and paste by ctrl+v into the Topic of 
        Concern box and Detail Problem Description Box. Hence, the Copy and Paste functionality is necessary.

5) Now upload your contract text or image in the Upload your legal document box. You can provide text input also.

6) Select the language of the contract text/image and submit

7) You will receive the desired output, referring to whether you have a contradiction in any sentences in the related paragraphs of the legal document.

8) Analyse the output and take further steps according to it.

9) Voice-based assistance according to user language will be provided at each step making the app more interactive and easier to use.

## Final Training and Eval loss for all languages

1.  Bn0.717600 | 0.812157
    
2.  Hi0.688700 | 0.785396
    
3.  Gu 0.745400 | 0.770488
    
4.  Pa 0.723800 | 0.764565
    
5.  Mr 0.677900 | 0.806058
    

## Resources

1) Web-parsed version of Sanskrit OCR using Selenium.

2) Semantic Search paraphrase-multilingual-mpnet-base-v2.3)      Transliteration API (Source: Bhashini)

4) Translation API (Source: Bhashini)

5) ASR API (Source: Bhashini)

6) Fine-tuning NLI model Deberta Base V3 for Hindi, Bengali, Gujarati, Marathi, and Punjabi from IndicXNLI

7) Pretrained cross-encoder NLI Deberta V3 base on SNLI and MNLI.
   
8) Pretrained MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 to support the English language.

9) Document text reader using PyMuPDF

All resources are available on the GitHub repository.

Note: prefer to use single-pretrained-model branch because of it being more responsive and lightweight than this one.
