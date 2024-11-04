# Car Reviews Analysis

This project performs various Natural Language Processing (NLP) tasks on car reviews using pre-trained models from the Hugging Face Transformers library. The tasks include sentiment analysis, translation, question answering, summarization, and toxicity analysis. This project was developed as part of a DataCamp online course.

## Steps

### Step 1: Sentiment Analysis
This script performs sentiment analysis on car reviews using a pre-trained model from the transformers library. It calculates and prints the accuracy and F1 score of the sentiment predictions.

### Step 2: Translation
This script translates car reviews from English to Spanish using a pre-trained translation model from the transformers library. It calculates and prints the BLEU score of the translation.

### Step 3: Question Answering
This script answers a question about a car review using a pre-trained question-answering model from the transformers library. It prints the answer to the specified question based on the context provided.

#### Solution 1
Uses the `pipeline` function for question answering.

#### Solution 2
Uses `AutoTokenizer` and `AutoModelForQuestionAnswering` for more control over the question-answering process.

### Step 4: Summarization and Toxicity Analysis
This script summarizes a car review using a pre-trained summarization model from the transformers library. It prints the summarized text and performs toxicity analysis on the summarized text.

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


