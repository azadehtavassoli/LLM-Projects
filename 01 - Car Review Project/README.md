# Car Reviews Analysis

This project performs various Natural Language Processing (NLP) tasks on car reviews using pre-trained models from the Hugging Face Transformers library. The tasks include sentiment analysis, translation, question answering, and summarization. These tasks demonstrate how language models can support customer-focused companies like "Car-ing is sharing" in managing customer feedback, translations, and information extraction. This project was developed as part of a DataCamp online course.

## Project Overview

The CTO of "Car-ing is sharing," a car sales and rental company, requested prototyping a chatbot app powered by large language models (LLMs) to address diverse inquiries. This project uses pre-trained LLMs to perform various NLP tasks on car reviews to simulate potential chatbot functionalities. Each task showcases a different aspect of LLM capabilities relevant to customer interactions.

### Key Tasks

1.  **Sentiment Analysis**  
    This task classifies the sentiment of car reviews to determine whether they are positive or negative. The results are then evaluated for accuracy and F1 score.
    
2.  **Translation**  
    To support the company's growing Spanish-speaking customer base, the first two sentences of a review are translated into Spanish using an English-to-Spanish translation model. A BLEU score is calculated to evaluate the translation quality.
    
3.  **Question Answering**  
    This task uses an extractive question-answering model to address specific inquiries related to brand aspects within a car review. For instance, given the question "What did they like about the brand?", the model provides relevant answers based on the review's content.
    
4.  **Summarization**  
    The last review is summarized into a concise format (approximately 50–55 tokens), providing a quick overview of the review.
    

## Steps

### Step 1: Sentiment Analysis

This script performs sentiment analysis on car reviews using a pre-trained model from the Transformers library. The sentiment predictions are then evaluated using accuracy and F1 score metrics.

-   **Implementation**:
    -   The script loads a dataset (`car_reviews.csv`) containing car reviews.
    -   Sentiment is predicted for each review, storing results in `predicted_labels`.
    -   The predictions are mapped to binary labels (0 for negative, 1 for positive) in `predictions`.
    -   The `accuracy_result` and `f1_result` variables store the calculated metrics for evaluation.

### Step 2: Translation

This script translates the first two sentences of the first car review into Spanish using a pre-trained translation model. The translation is evaluated using BLEU score.

-   **Implementation**:
    -   The model extracts the first two sentences from the first review and translates them to Spanish.
    -   The translated output is stored in `translated_review`.
    -   Using a reference translation (`reference_translations.txt`), the script calculates the BLEU score and stores it in `bleu_score`.

### Step 3: Question Answering

This script answers a brand-related question within a car review, demonstrating the model's capability to identify brand sentiments or aspects mentioned by the reviewer.

-   **Implementation**:
    -   The model loads the context from the second review and uses an extractive question-answering model to answer the question, "What did they like about the brand?"
    -   The `question` and `context` variables hold the input text for the model.
    -   The answer is stored in the `answer` variable.

### Step 4: Summarization

This script summarizes the last review, providing a brief overview of its content.

-   **Implementation**:
    -   A pre-trained summarization model reduces the last review in the dataset to approximately 50–55 tokens.
    -   The summarized content is stored in `summarized_text`.

## Requirements
- Python
- Transformers library
- NLTK
- Pandas
- Scikit-learn
- Evaluate library

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/car-reviews-analysis.git
   ```
2. Navigate to the project directory:
   ```sh
   cd car-reviews-analysis
   ```
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the script:
```sh
python carreview.py
```

## License
This project is licensed under the MIT License.


