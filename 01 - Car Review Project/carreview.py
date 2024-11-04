"""
Step1: This script performs sentiment analysis on car reviews using a pre-trained model from the transformers library.
It calculates and prints the accuracy and F1 score of the sentiment predictions.
"""
from transformers import logging, pipeline
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, log_loss

logging.set_verbosity(logging.WARNING)  # Set logging level to WARNING to suppress unnecessary messages
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Disable symlink warning for Hugging Face Hub

# Load the car reviews dataset
df = pd.read_csv('data/car_reviews.csv', sep=';')
print("First 10 lines of the DataFrame:")
print(df.head(10))  # Print the first 10 lines of the DataFrame

# Initialize a sentiment analysis pipeline with a pre-trained model
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Define an example text to analyze
example_text = "I love this car! It's amazing."

# Perform sentiment analysis on the example text
result = classifier(example_text)
print("Sentiment analysis result for example text:")
print(result)

# Perform sentiment analysis on the reviews
predicted_labels = classifier(df['Review'].tolist())

# Extract labels and map to {0,1}
predictions = [1 if label['label'] == 'POSITIVE' else 0 for label in predicted_labels]

# Map actual labels to {0,1}
actual_labels = df['Class'].apply(lambda x: 1 if x == 'POSITIVE' else 0).tolist()

# Calculate metrics
accuracy_result = accuracy_score(actual_labels, predictions)
f1_result = f1_score(actual_labels, predictions)

# Print the calculated metrics
print(f"Accuracy: {accuracy_result}, F1 Score: {f1_result}")

"""
Step2: This script translates car reviews from English to Spanish using a pre-trained translation model from the transformers library.
It calculates and prints the BLEU score of the translation.
"""
import nltk
nltk.download('punkt')  # Download the 'punkt' tokenizer models from NLTK

# Initialize the translation pipeline with a pre-trained model
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

# Translate the first review from English to Spanish
first_review = df['Review'][0]
translated_review = translator(first_review, max_length=55)[0]['translation_text']
print(f"Model translation:\n{translated_review}")

# Load reference translations from the text file
with open("data/reference_translations.txt", "r") as file:
    lines = file.read().splitlines()
references = [line.strip() for line in lines]
print(f"Spanish translation references:\n{references}")

# Load and calculate BLEU score metric
import evaluate
bleu = evaluate.load("bleu")
bleu_score = bleu.compute(predictions=[translated_review], references=[references])
#print(bleu_score['bleu'])

"""
Step3: Solution1: This script answers a question about a car review using a pre-trained question-answering model from the transformers library.
It prints the answer to the specified question based on the context provided.
"""
qa_pipeline = pipeline("question-answering", model="deepset/minilm-uncased-squad2")
context = df['Review'][1]
print("context:")
print(context)
question = "What did he like about the brand?"
answer = qa_pipeline(question=question, context=context)['answer']
print(f"Answer: {answer}")

"""
Step3: Solution2: This script answers a question about a car review using a pre-trained question-answering model from the transformers library.
It prints the answer to the specified question based on the context provided.
"""
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Instantiate model and tokenizer
model_ckp = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckp)
model = AutoModelForQuestionAnswering.from_pretrained(model_ckp)

# Define context and question, then tokenize them
context = df['Review'][1]
question = "What did he like about the brand?"
inputs = tokenizer(question, context, return_tensors="pt")

# Perform inference and extract answer from raw model outputs
with torch.no_grad():
    outputs = model(**inputs)

start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits) + 1
answer_span = inputs["input_ids"][0][start_idx:end_idx]
answer = tokenizer.decode(answer_span)
#print("Answer:", answer)

"""
Step4:This script summarizes a car review using a pre-trained summarization model from the transformers library.
It prints the summarized text and performs toxicity analysis on the summarized text.
"""
# Check the last review
last_review = df['Review'].iloc[-1]
print("Last review:")
print(last_review)

# Summarize the last review
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarized_text = summarizer(last_review, max_length=55, min_length=50, do_sample=False)[0]['summary_text']
print("Summarized text:")
print(summarized_text)

# Code for Toxicity Analysis
toxicity_analyzer = pipeline("text-classification", model="unitary/toxic-bert")
summarized_text = (summarized_text[:200])
toxicity_result = toxicity_analyzer(summarized_text)
print(f"Summarized Text: {summarized_text}")
print("Toxicity Analysis:", toxicity_result)