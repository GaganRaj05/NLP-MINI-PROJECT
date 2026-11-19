# NLP Pipeline Lab

NLP Pipeline Lab is an interactive Streamlit application that
demonstrates a practical Natural Language Processing (NLP) workflow.
Users can enter text and inspect how it moves through tokenization,
cleaning, normalization, linguistic analysis, and vectorization.

The application is designed as an educational lab: each stage exposes
intermediate outputs so users can compare techniques and understand how
raw text is transformed into features that can be used by
machine-learning models.

## Features

-   **Tokenization:** Compare Python `split()`, NLTK `word_tokenize`,
    and spaCy tokenization.
-   **Text cleaning:** Lowercase text, remove emoji, URLs, email
    addresses, and non-letter characters while retaining apostrophes.
-   **Stop-word filtering:** Inspect tokens and see which English stop
    words are removed.
-   **Normalization:** Compare Porter stemming, Lancaster stemming, NLTK
    WordNet lemmatization, and spaCy lemmatization.
-   **Part-of-speech tagging:** View token-level POS tags, fine-grained
    tags, and tag descriptions.
-   **Named entity recognition:** Extract entities using spaCy and
    display their labels.
-   **Vectorization:** Generate Bag of Words and TF-IDF matrices from
    the input's sentences.
-   **Word embeddings:** Optionally train a small Word2Vec model on the
    entered text and inspect a selected word's 50-dimensional vector.
-   **Interactive exploration:** Use sample demo text, clear the input,
    and explore each stage through dedicated tabs.
-   **Caching:** Cache language resources and pipeline computations to
    avoid repeating expensive processing unnecessarily.

## NLP Pipeline

``` text
             Input Text
                 |
                 v
            Tokenization
                 |
                 v
          Text Cleaning
                 |
                 v
           Normalization
       (Stemming / Lemmatization)
                 |
                 v
       Linguistic Processing
       (POS Tags / NER)
                 |
                 v
            Vectorization
        (BoW / TF-IDF / Word2Vec)
```

## Application Sections

The Streamlit interface contains the following tabs:

  -----------------------------------------------------------------------
  Tab                                 What it demonstrates
  ----------------------------------- -----------------------------------
  Overview                            Text statistics, a sentence-level
                                      transformation journey, and
                                      practical NLP use cases

  Tokenization                        Comparison of Python, NLTK, and
                                      spaCy tokenizers

  Cleaning                            Text cleaning output and stop-word
                                      filtering

  Normalization                       Porter, Lancaster, NLTK WordNet,
                                      and spaCy lemma comparisons

  Linguistics                         POS tagging, POS distribution, and
                                      named entity recognition

  Vectorization                       Bag of Words, TF-IDF, and optional
                                      Word2Vec vectors

  Comparisons                         Tokenizer comparison and
                                      stemming-versus-lemmatization
                                      comparison
  -----------------------------------------------------------------------

## Technology Stack

  Technology     Purpose
  -------------- ---------------------------------------------------------------
  Python         Application and NLP processing
  Streamlit      Interactive web application
  NLTK           Tokenization, stop words, stemming, and WordNet lemmatization
  spaCy          POS tagging, named entity recognition, and lemmatization
  scikit-learn   Bag of Words and TF-IDF vectorization
  gensim         Optional Word2Vec training
  pandas         Tabular NLP outputs
  emoji          Emoji removal during text cleaning

## Project Structure

``` text
NLP-Pipeline-Lab/
├── app.py
├── requirements.txt
└── README.md
```

The main application code is in `app.py`. The application downloads the
required NLTK data packages and the spaCy English model when they are
missing, then caches the loaded language resources.

## Setup and Installation

### Prerequisites

-   Python 3.9 or a compatible version supported by the installed
    dependencies
-   pip

### 1. Clone or open the project

``` bash
git clone <YOUR_REPOSITORY_URL>
cd NLP-Pipeline-Lab
```

If you already have the project locally, open a terminal in the project
directory.

### 2. Create a virtual environment

**Windows**

``` powershell
python -m venv env
.\env\Scripts\activate
```

**Linux / macOS**

``` bash
python3 -m venv env
source env/bin/activate
```

### 3. Install dependencies

Create a `requirements.txt` file with:

``` text
streamlit
numpy
pandas
emoji
nltk
spacy
scikit-learn
gensim
```

Then install the packages:

``` bash
pip install -r requirements.txt
```

The application uses NLTK resources (`punkt_tab`, `punkt`, `stopwords`,
and `wordnet`) and the spaCy model `en_core_web_sm`. The application
attempts to download these automatically on first use if they are not
already installed. An internet connection may therefore be required on
the first run.

### 4. Run the Streamlit application

From the project directory, with the virtual environment activated:

``` bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, normally:

``` text
http://localhost:8501
```

Open that address in your browser.

## How to Use

1.  Open the application in your browser.
2.  Enter or paste text into the input area.
3.  Select **Run pipeline** to process the text.
4.  Use **Use demo text** to load the built-in example.
5.  Use **Clear** to reset the input and results.
6.  Explore the Overview, Tokenization, Cleaning, Normalization,
    Linguistics, Vectorization, and Comparisons tabs.
7.  In the Vectorization tab, enable **Train Word2Vec on these
    sentences** to train and inspect a small Word2Vec model on the
    current input.

## Processing Details

### Tokenization

The application compares three approaches:

-   Python `str.split()` separates text on whitespace.
-   NLTK `word_tokenize()` separates punctuation and handles
    contractions.
-   spaCy's tokenizer applies language-aware tokenization rules.

### Cleaning

The cleaning function:

1.  Converts text to lowercase.
2.  Removes emoji.
3.  Removes URLs and email addresses.
4.  Keeps alphabetic characters, whitespace, and apostrophes.
5.  Collapses repeated whitespace.

The resulting text is tokenized, and non-alphabetic tokens are excluded
from the subsequent token lists.

### Normalization

For each unique word, the application compares:

-   Porter stemmer
-   Lancaster stemmer
-   NLTK WordNet lemmatizer
-   spaCy lemmatizer

The NLTK and spaCy lemmatization paths use part-of-speech information
where available.

### Linguistic Analysis

spaCy processes the input to provide:

-   Sentence segmentation
-   Part-of-speech tags
-   Fine-grained POS tags and descriptions
-   Named entities and entity labels

### Vectorization

Each sentence is treated as a document. The application removes
non-alphabetic tokens and English stop words before creating:

-   **Bag of Words (BoW):** term-frequency counts.
-   **TF-IDF:** term-frequency/inverse-document-frequency values.

Word2Vec is optional and is trained only on the processed sentences
supplied by the user. Its vectors are illustrative and are not
pretrained general-purpose embeddings.

## Practical NLP Use Cases

The application includes guidance for several example use cases:

-   **Sentiment Analysis:** Preserve negations and adjectives; TF-IDF
    with a linear classifier is a useful baseline.
-   **Resume Screening:** Identify skills, organizations, and places,
    then compare candidate text with job descriptions.
-   **Chatbot Understanding:** Extract intent-relevant terms and
    entities that can fill slots such as dates and places.
-   **Spam Detection:** URLs, email addresses, and emoji can be useful
    signals, so extract them before removing them.
-   **Search Engine Processing:** Normalize terms so different word
    forms can match indexed content.

These are educational explanations in the interface; the application
itself demonstrates NLP preprocessing and feature extraction rather than
training a sentiment, resume-screening, chatbot, spam, or search-ranking
classifier.

## Notes and Limitations

-   The application is an interactive NLP demonstration, not a complete
    production NLP system.
-   Word2Vec is trained on the user's current input only. The resulting
    vectors are for illustration and will be limited when the input
    contains few sentences or words.
-   The first run may take longer because language resources are loaded
    and missing resources may be downloaded.
-   The app processes English text using the configured English
    stop-word list and spaCy model.
-   Vectorization requires enough meaningful text after cleaning and
    stop-word removal. Very short or heavily filtered input may not
    produce a feature matrix.

## Disclaimer

This project is intended for educational purposes and for demonstrating
common NLP preprocessing, linguistic analysis, and text vectorization
techniques.
