# Fine-Tune-BERT-Transformer-with-Spacy-3-for-NER

## Quick Intro
In this github repo, I will show how to train a BERT Transformer for Name Entity Recognition task using the latest Spacy 3 library. I am going to train an NER classifier to extract entities from scientific abstracts. 

Note: Fine tuning transformers requires a powerful GPU with parallel processing. For this I am using Google Colab since it provides freely available servers with GPUs.

<b> So, question may arise in your mind, What is BERT transformer? </b>

BERT stands for Bidirectional Encoder Representations from Transformers— leverages the transformer architecture in a novel way. For example, BERT analyses both sides of the sentence with a randomly masked word to make a prediction. In addition to predicting the masked token, BERT predicts the sequence of the sentences by adding a classification token [CLS] at the beginning of the first sentence and tries to predict if the second sentence follows the first one by adding a separation token[SEP] between the two sentences.

## Dataset or Data Labeling for Train and Test Set

_**Dataset Requirement**_
* Training and dev data in the spaCy 3 JSON Format 
  * Step 01: First, Generate IOB format in .tsv format using UBIAI Text Annotation Tool
  * Step 02: Convert to SpaCy JSON Format 

_**Note**:_ The training data and testing data was obtained using the **UBIAI** Text Annotation Tool.

You might get confused about **UBIAI**, following are the features about UBIAI Text Annotation Tool:

* ML auto-annotation
* Dictionary, regex, and rule-based auto-annotation
* Team collaboration to share annotation tasks
* Direct annotation export to IOB format

