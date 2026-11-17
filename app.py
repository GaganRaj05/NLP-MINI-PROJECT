import streamlit as st
import re
import emoji
import nltk
import spacy
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec 

# ================= NLP DOWNLOADS =================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="🧠 NLP Wonder Dashboard",
    layout="wide",
    page_icon="🧠"
)

# ================= SESSION STATE =================
if "text" not in st.session_state:
    st.session_state.text = ""

if "run_pipeline" not in st.session_state:
    st.session_state.run_pipeline = False

# ================= SIDEBAR =================
st.sidebar.title("🧪 NLP Control Panel")

st.sidebar.markdown("### 📌 Libraries Used")
st.sidebar.markdown("""
- **NLTK** → Tokenization, Stopwords, Stemming  
- **spaCy** → POS Tagging, NER  
- **scikit-learn** → BoW, TF-IDF  
- **gensim** → Word2Vec  
""")

if st.sidebar.button("📄 Load Demo Text"):
    st.session_state.text = """
Great question, Ahmed! Since you're in study mode, let's explore NLP properly.
I am learning NLP in Hyderabad and building AI projects using Python.
Yesterday I emailed ahmed.faheen@gmail.com and visited https://openai.com 😊.
"""

if st.sidebar.button("🔄 Reset"):
    st.session_state.text = ""
    st.session_state.run_pipeline = False

use_case = st.sidebar.selectbox(
    "🎯 Real-World Use Case",
    [
        "Sentiment Analysis",
        "Resume Screening",
        "Chatbot Understanding",
        "Spam Detection",
        "Search Engine Processing"
    ]
)

# ================= HEADER =================
st.title("🧠 NLP Wonder Dashboard")
st.caption("Explore the complete NLP pipeline — interactively, visually, professionally")

# ================= CHAT INPUT =================
st.divider()
user_input = st.chat_input("Type or paste text here and press Enter ↗️")

if user_input:
    st.session_state.text = user_input
    st.session_state.run_pipeline = True

# ================= SUBMIT / CLEAR =================
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 Submit"):
        st.session_state.run_pipeline = True

with col2:
    if st.button("🧹 Clear"):
        st.session_state.text = ""
        st.session_state.run_pipeline = False

text = st.session_state.text

# ================= PREPROCESSING =================
def clean_text(txt):
    txt = txt.lower()
    txt = emoji.replace_emoji(txt, replace="")
    txt = re.sub(r"http\S+|www\S+|\S+@\S+", "", txt)
    txt = re.sub(r"[^a-z\s]", "", txt)
    return txt.strip()

cleaned_text = clean_text(text)
tokens = word_tokenize(cleaned_text) if cleaned_text else []
filtered_words = [w for w in tokens if w not in stop_words]

# ================= STATUS BAR =================
if st.session_state.run_pipeline and text.strip():
    st.success("✅ Pipeline executed successfully")
elif not text.strip():
    st.info("ℹ️ Enter text to start the NLP pipeline")

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📌 Overview",
    "🔤 Tokenization",
    "🧹 Cleaning",
    "🌱 Normalization",
    "🧠 Linguistics",
    "📊 Vectorization",
    "⚖️ Comparisons"
])

# =========================================================
# TAB 1 – OVERVIEW
# =========================================================
with tab1:
    st.subheader("🔁 NLP Pipeline Flow")

    st.code("""
Raw Text
 ↓
Tokenization (NLTK / spaCy)
 ↓
Text Cleaning
 ↓
Stemming / Lemmatization
 ↓
POS Tagging & NER (spaCy)
 ↓
Vectorization (ML Ready)
""")

    st.subheader("💼 Real-World Application Flow")
    st.code(f"""
User Input
 ↓
NLP Preprocessing
 ↓
Feature Engineering
 ↓
ML / AI Model
 ↓
Prediction
({use_case})
""")

# =========================================================
# TAB 2 – TOKENIZATION
# =========================================================
with tab2:
    st.subheader("🔤 Tokenization Lab")
    st.caption("Compare how different libraries tokenize text")

    if st.session_state.run_pipeline:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Python split()**")
            st.caption("❌ Not NLP-aware")
            st.write(text.split())

        with col2:
            st.markdown("**NLTK Tokenizer**")
            st.caption("📦 nltk.tokenize.word_tokenize")
            st.write(word_tokenize(text))

        with col3:
            st.markdown("**spaCy Tokenizer**")
            st.caption("⚡ spaCy language-aware tokenizer")
            st.write([t.text for t in nlp(text)])

# =========================================================
# TAB 3 – CLEANING
# =========================================================
with tab3:
    st.subheader("🧹 Text Cleaning")
    st.caption("Lowercasing, emoji, URL, email & special character removal")

    if st.session_state.run_pipeline:
        st.write(cleaned_text)

# =========================================================
# TAB 4 – NORMALIZATION
# =========================================================
with tab4:
    st.subheader("🌱 Normalization")
    st.caption("Reducing words to base form")

    if len(tokens) == 0:
        st.warning("Not enough words to normalize.")
    else:
        porter = PorterStemmer()
        lancaster = LancasterStemmer()
        lemmatizer = WordNetLemmatizer()

        df = pd.DataFrame({
            "Original": tokens,
            "Porter Stemmer (NLTK)": [porter.stem(w) for w in tokens],
            "Lancaster Stemmer (NLTK)": [lancaster.stem(w) for w in tokens],
            "Lemmatizer (NLTK)": [lemmatizer.lemmatize(w, pos="v") for w in tokens]
        })

        st.dataframe(df, use_container_width=True)

# =========================================================
# TAB 5 – LINGUISTICS
# =========================================================
with tab5:
    st.subheader("🧠 Linguistic Analysis")
    st.caption("Powered by spaCy")

    if not text.strip():
        st.warning("Please enter text.")
    else:
        doc = nlp(text)

        with st.expander("🔹 POS Tagging (spaCy)"):
            st.dataframe(
                pd.DataFrame(
                    [(t.text, t.pos_) for t in doc],
                    columns=["Word", "POS Tag"]
                ),
                use_container_width=True
            )

        with st.expander("🔹 Named Entity Recognition (spaCy)"):
            st.write([(ent.text, ent.label_) for ent in doc.ents])

# =========================================================
# TAB 6 – VECTORIZATION
# =========================================================
with tab6:
    st.subheader("📊 Vectorization")
    st.caption("Convert text into numbers (ML-ready)")

    if len(filtered_words) == 0:
        st.warning("Not enough meaningful words.")
    else:
        corpus = [" ".join(filtered_words)]

        with st.expander("🔹 Bag of Words (scikit-learn)"):
            bow = CountVectorizer()
            bow_matrix = bow.fit_transform(corpus)
            st.dataframe(
                pd.DataFrame(
                    bow_matrix.toarray(),
                    columns=bow.get_feature_names_out()
                ),
                use_container_width=True
            )

        with st.expander("🔹 TF-IDF (scikit-learn)"):
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(corpus)
            st.dataframe(
                pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=tfidf.get_feature_names_out()
                ),
                use_container_width=True
            )

        with st.expander("🔹 Word2Vec (gensim)"):
            model = Word2Vec(
                [filtered_words],
                vector_size=50,
                window=5,
                min_count=1
            )
            st.write(model.wv[filtered_words[0]])

# =========================================================
# TAB 7 – COMPARISONS
# =========================================================
with tab7:
    st.subheader("⚖️ Method Comparison")

    st.table(pd.DataFrame({
        "Method": ["Python split()", "NLTK", "spaCy"],
        "Language Aware": ["❌", "⚠️", "✅"],
        "Handles Punctuation": ["❌", "✅", "✅"],
        "Production Ready": ["❌", "⚠️", "✅"]
    }))

    st.markdown("### 🌱 Stemming vs Lemmatization")
    st.table(pd.DataFrame({
        "Aspect": ["Accuracy", "Speed", "Meaning Preserved"],
        "Stemming": ["Low", "Fast", "❌"],
        "Lemmatization": ["High", "Slower", "✅"]
    }))
