import html
import re
from types import SimpleNamespace

import emoji
import pandas as pd
import streamlit as st


st.set_page_config(page_title="NLP Pipeline Lab", page_icon="🧠", layout="wide")

C = {"tok": "#3b6cf6", "clean": "#0fa89a", "norm": "#3f9e4d", "ling": "#e8920c", "vec": "#8a5cf0"}

DEMO_TEXT = (
    "Great question, Gagan! Since you're in study mode, let's explore NLP properly. "
    "I am learning NLP in Mangalore and building AI projects using Python. "
    "Yesterday I emailed gaganraj.dev05@gmail.com 😊."
)

USE_CASES = {
    "Sentiment Analysis": ("Label text as positive, negative or neutral.",
                           "Keep negations and adjectives. TF-IDF with a linear classifier is a strong baseline."),
    "Resume Screening": ("Match candidate skills against a job description.",
                         "Named entities (organisations, skills, places) and TF-IDF similarity do most of the work."),
    "Chatbot Understanding": ("Work out what the user wants and which details matter.",
                              "Keep stop words for intent models, and use entities to fill slots such as dates and places."),
    "Spam Detection": ("Flag unwanted messages before they reach the inbox.",
                       "URLs, emails and emoji are signals here, so extract them before you strip them."),
    "Search Engine Processing": ("Turn a query into terms that match indexed documents.",
                                 "Lowercase, drop stop words and lemmatize so 'running' finds 'run'."),
}

STAGES = [
    ("Tokenizing", "tok"), ("Cleaning", "clean"), ("Normalizing", "norm"),
    ("Tagging", "ling"), ("Vectorizing", "vec"),
]

LIBS_NOTE = ("Built with NLTK (tokens, stop words, stemming), spaCy (tags, entities), "
             "scikit-learn (Bag of Words, TF-IDF) and gensim (Word2Vec).")

WN_POS = {"NOUN": "n", "PROPN": "n", "VERB": "v", "AUX": "v", "ADJ": "a", "ADV": "r"}

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Figtree:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root { --ink:#171a2b; --muted:#4a5170; --line:#e1e4f0; --paper:#f4f5fa; --surface:#fff;
  --tok:#3b6cf6; --clean:#0fa89a; --norm:#3f9e4d; --ling:#e8920c; --vec:#8a5cf0; }

html, body, .stApp, .stMarkdown, button, input, textarea, label { font-family:'Figtree', sans-serif !important; }
.stApp { background: var(--paper); color: var(--ink); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1160px; padding-top: 2.6rem; padding-bottom: 4rem; }

/* headings */
h1.lab-title { font-family:'Bricolage Grotesque', sans-serif !important; font-weight:700; font-size:3rem;
  letter-spacing:-0.035em; line-height:1; margin:0; color:var(--ink); }
p.lab-sub { color:var(--muted); font-size:1.08rem; margin:.7rem 0 0; max-width:58ch; line-height:1.5; }
h2, h3, h4 { font-family:'Bricolage Grotesque', sans-serif !important; letter-spacing:-0.02em; color:var(--ink); }
.dots { display:flex; gap:.4rem; margin-top:1.1rem; }
.dots i { width:34px; height:6px; border-radius:99px; display:block; }

/* input panel */
div[data-testid="stVerticalBlockBorderWrapper"] { background:var(--surface); border:1px solid var(--line) !important;
  border-radius:14px; box-shadow:0 1px 0 rgba(23,26,43,.03), 0 12px 32px -20px rgba(59,108,246,.35); }
.stTextArea textarea { font-size:1.02rem; line-height:1.55; border-radius:10px; border:1px solid var(--line);
  background:#fbfbfe; color:var(--ink); }
.stTextArea textarea::placeholder { color:#6b7290 !important; opacity:1; }
.stTextArea textarea:focus { border-color:var(--tok); box-shadow:0 0 0 3px rgba(59,108,246,.15); }

/* buttons: readable labels in every state (the label sits inside a <p>) */
.stButton > button { border-radius:10px; font-weight:600; padding:.45rem 1.1rem; border:1px solid var(--line);
  background:#fff; transition:transform .12s ease, box-shadow .12s ease; }
.stButton > button,
.stButton > button p { color:var(--ink) !important; }
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 6px 16px -8px rgba(23,26,43,.35); border-color:var(--tok); }
.stButton > button:hover,
.stButton > button:hover p { color:var(--tok) !important; }

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] { background:var(--ink); border-color:var(--ink); }
.stButton > button[kind="primary"] p,
.stButton > button[data-testid="stBaseButton-primary"] p { color:#fff !important; }
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover { background:var(--tok); border-color:var(--tok); }
.stButton > button[kind="primary"]:hover p,
.stButton > button[data-testid="stBaseButton-primary"]:hover p { color:#fff !important; }

/* disabled "Run pipeline": legible, but visibly inactive */
.stButton > button:disabled,
.stButton > button[kind="primary"]:disabled,
.stButton > button[data-testid="stBaseButton-primary"]:disabled { opacity:1; background:#e6e8f2; border-color:#e6e8f2;
  transform:none; box-shadow:none; }
.stButton > button:disabled p,
.stButton > button[kind="primary"]:disabled p,
.stButton > button[data-testid="stBaseButton-primary"]:disabled p { color:#6b7290 !important; }

/* tabs: each stage has its own colour dot */
.stTabs [data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { height:2.8rem; padding:0 .9rem; font-weight:500; color:var(--muted); }
.stTabs [aria-selected="true"] { color:var(--ink) !important; font-weight:600; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--ink) !important; height:2px; }
.stTabs button[role="tab"]::before { content:""; display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:.55rem; background:#b9bed3; }
.stTabs button[role="tab"]:nth-of-type(2)::before { background:var(--tok); }
.stTabs button[role="tab"]:nth-of-type(3)::before { background:var(--clean); }
.stTabs button[role="tab"]:nth-of-type(4)::before { background:var(--norm); }
.stTabs button[role="tab"]:nth-of-type(5)::before { background:var(--ling); }
.stTabs button[role="tab"]:nth-of-type(6)::before { background:var(--vec); }

/* stat tiles */
.stats { display:grid; grid-template-columns:repeat(5, 1fr); gap:.8rem; margin:1.2rem 0 1.6rem; }
.stat { background:var(--surface); border:1px solid var(--line); border-top:3px solid var(--c); border-radius:10px; padding:.9rem 1rem; }
.stat b { display:block; font-family:'Bricolage Grotesque', sans-serif; font-size:2rem; line-height:1.1; letter-spacing:-0.03em; color:var(--ink); }
.stat span { color:var(--muted); font-size:.88rem; }
@media (max-width: 760px) { .stats { grid-template-columns:repeat(2, 1fr); } }

/* chips */
.chips { display:flex; flex-wrap:wrap; gap:.35rem; margin:.5rem 0 1rem; }
.chip { font-family:'IBM Plex Mono', monospace; font-size:.8rem; padding:.2rem .55rem; border-radius:6px;
  color:var(--ink); background:color-mix(in srgb, var(--c, #8890ad) 13%, white); border:1px solid color-mix(in srgb, var(--c, #8890ad) 35%, white); }
.chip.muted { background:#eceef6; border-color:#dcdfec; color:#6b7290; text-decoration:line-through; }
.chip b { font-family:'Figtree', sans-serif; font-size:.68rem; font-weight:600; margin-left:.5rem; color:var(--c); }

/* journey rows */
.row { display:grid; grid-template-columns:130px 1fr; gap:1rem; padding:.85rem 0; border-bottom:1px dashed var(--line); align-items:start; }
.row:last-child { border-bottom:0; }
.row .lab { display:flex; align-items:center; gap:.55rem; font-weight:600; font-size:.95rem; padding-top:.2rem; color:var(--ink); }
.row .lab i { width:10px; height:10px; border-radius:50%; background:var(--c); display:block; }
.row .val { line-height:1.6; color:var(--ink); }
.row .chips { margin:0; }
.panel { background:var(--surface); border:1px solid var(--line); border-left:4px solid var(--c); border-radius:10px;
  padding:.9rem 1.1rem; font-family:'IBM Plex Mono', monospace; font-size:.86rem; line-height:1.65; white-space:pre-wrap;
  word-break:break-word; color:var(--ink); }

/* flow + notes */
.flow { display:flex; flex-wrap:wrap; align-items:center; gap:.45rem; margin:.5rem 0 1.4rem; }
.flow .step { padding:.4rem .85rem; border-radius:99px; border:1px solid var(--line); background:var(--surface); font-weight:500; font-size:.9rem; color:var(--ink); }
.flow .step.last { background:var(--ink); color:#fff; border-color:var(--ink); }
.flow .arrow { color:#8890ad; }
.note { background:var(--surface); border:1px solid var(--line); border-left:4px solid var(--c, #b9bed3); border-radius:10px; padding:.8rem 1rem; color:var(--muted); line-height:1.55; margin:.6rem 0 1rem; }

/* empty state */
.empty { display:grid; grid-template-columns:repeat(5, 1fr); gap:.8rem; margin-top:1.4rem; }
.empty div { background:var(--surface); border:1px solid var(--line); border-top:3px solid var(--c); border-radius:10px; padding:1rem; }
.empty b { display:block; margin-bottom:.35rem; color:var(--ink); font-size:1rem; }
.empty span { color:var(--muted); font-size:.88rem; line-height:1.45; display:block; }
@media (max-width: 900px) { .empty { grid-template-columns:1fr 1fr; } }

/* loader */
.loader { background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:1.3rem 1.5rem; margin:1.2rem 0;
  box-shadow:0 12px 32px -20px rgba(59,108,246,.4); }
.loader h4 { margin:0 0 .2rem; font-size:1.15rem; }
.loader p { margin:0 0 1rem; color:var(--muted); font-size:.92rem; }
.bar { height:6px; border-radius:99px; background:#eceef6; overflow:hidden; margin-bottom:1.1rem; }
.bar i { display:block; height:100%; border-radius:99px; background:linear-gradient(90deg, var(--tok), var(--vec));
  background-size:200% 100%; animation:slide 1.4s linear infinite; transition:width .35s ease; }
.ls { display:flex; flex-wrap:wrap; gap:.5rem 1.6rem; }
.ls div { display:flex; align-items:center; gap:.55rem; font-weight:500; color:#8890ad; }
.ls .dot { width:16px; height:16px; border-radius:50%; border:2px solid #d5d9ea; box-sizing:border-box; }
.ls .done { color:var(--ink); }
.ls .done .dot { background:var(--c); border-color:var(--c); }
.ls .active { color:var(--ink); }
.ls .active .dot { border-color:color-mix(in srgb, var(--c) 25%, white); border-top-color:var(--c); animation:spin .7s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
@keyframes slide { to { background-position:-200% 0; } }
@media (prefers-reduced-motion: reduce) { .bar i, .ls .active .dot { animation:none; } }

[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- resources (loaded on first run)
@st.cache_resource(show_spinner="Loading language models. This only happens once.")
def res():
    import nltk
    import spacy
    from nltk.corpus import stopwords
    from nltk.stem import LancasterStemmer, PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    for pkg, path in [
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("punkt", "tokenizers/punkt"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    lemma = WordNetLemmatizer()
    lemma.lemmatize("warmup")      # WordNet loads lazily; pay that cost now, once
    word_tokenize("warm up.")      # same for the punkt model
    return SimpleNamespace(
        nlp=nlp, stop=set(stopwords.words("english")), tokenize=word_tokenize,
        porter=PorterStemmer(), lancaster=LancasterStemmer(), lemma=lemma, explain=spacy.explain,
    )


# ---------------------------------------------------------------- helpers
def chips(items, color=None, cls=""):
    if not items:
        return "<div class='note'>Nothing to show.</div>"
    style = f" style='--c:{color}'" if color else ""
    body = "".join(f"<span class='chip {cls}'{style}>{html.escape(str(i))}</span>" for i in items)
    return f"<div class='chips'>{body}</div>"


def show_df(data, **kw):
    try:
        st.dataframe(data, width="stretch", hide_index=True, **kw)
    except TypeError:  # older Streamlit
        st.dataframe(data, use_container_width=True, hide_index=True, **kw)


def clean_text(txt):
    txt = txt.lower()
    txt = emoji.replace_emoji(txt, replace="")
    txt = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", txt)
    txt = re.sub(r"[^a-z\s']", " ", txt)  # keep apostrophes so "you're" survives tokenization
    return re.sub(r"\s+", " ", txt).strip()


def stat(label, value, color):
    return f"<div class='stat' style='--c:{color}'><b>{value}</b><span>{label}</span></div>"


def loader_html(active):
    done = active / len(STAGES)
    steps = "".join(
        f"<div class='{'done' if i < active else 'active' if i == active else ''}' style='--c:{C[k]}'>"
        f"<span class='dot'></span>{label}</div>"
        for i, (label, k) in enumerate(STAGES)
    )
    return (
        "<div class='loader'><h4>Running the pipeline</h4>"
        "<p>Your text is moving through each stage.</p>"
        f"<div class='bar'><i style='width:{max(done * 100, 6):.0f}%'></i></div>"
        f"<div class='ls'>{steps}</div></div>"
    )


def lemma_of(lex, word):
    """(spaCy lemma, POS) for a lowercase word, reusing the already parsed document."""
    hit = lex.get(word)
    if hit:
        return hit
    t = res().nlp(word)[0]
    return t.lemma_.lower(), t.pos_


# ---------------------------------------------------------------- pipeline stages (cached)
@st.cache_data(show_spinner=False)
def s_spacy(text):
    """The single spaCy pass. Every other stage reuses this instead of re-parsing."""
    R = res()
    doc = R.nlp(text)
    lex = {}
    for t in doc:
        lex.setdefault(t.text.lower(), (t.lemma_.lower(), t.pos_))
    return {
        "sentences": [s.text.strip() for s in doc.sents if s.text.strip()],
        "pos": pd.DataFrame(
            [(t.text, t.pos_, t.tag_, R.explain(t.tag_) or "") for t in doc if not t.is_space],
            columns=["Token", "POS", "Fine tag", "Meaning"]),
        "ents": [(e.text, e.label_, R.explain(e.label_) or "") for e in doc.ents],
        "lex": lex,
    }


@st.cache_data(show_spinner=False)
def s_tokenize(text):
    R = res()
    return {"split": text.split(), "nltk": R.tokenize(text),
            "spacy": [t.text for t in R.nlp.tokenizer(text)]}  # tokenizer only, no full pipeline


@st.cache_data(show_spinner=False)
def s_clean(text):
    R = res()
    cleaned = clean_text(text)
    tokens = [w for w in R.tokenize(cleaned) if w.isalpha()]
    return {"cleaned": cleaned, "tokens": tokens,
            "filtered": [w for w in tokens if w not in R.stop],
            "sentences": s_spacy(text)["sentences"]}


@st.cache_data(show_spinner=False)
def s_normalize(text):
    R = res()
    lex = s_spacy(text)["lex"]
    tokens = [w for w in R.tokenize(clean_text(text)) if w.isalpha()]
    rows = []
    for w in dict.fromkeys(tokens):
        sp_lemma, pos = lemma_of(lex, w)
        rows.append({
            "Original": w,
            "Porter": R.porter.stem(w),
            "Lancaster": R.lancaster.stem(w),
            "NLTK lemma": R.lemma.lemmatize(w, pos=WN_POS.get(pos, "n")),
            "spaCy lemma": sp_lemma,
        })
    return {"norm": pd.DataFrame(rows)}


@st.cache_data(show_spinner=False)
def s_tag(text):
    s = s_spacy(text)
    return {"pos": s["pos"], "ents": s["ents"]}


@st.cache_data(show_spinner=False)
def s_vectorize(sentences):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    R = res()
    docs = [d for d in (" ".join(w for w in R.tokenize(clean_text(s)) if w.isalpha() and w not in R.stop)
                        for s in sentences) if d]
    out = {"docs": docs, "bow": None, "tfidf": None}
    if docs:
        idx = [f"Sentence {i + 1}" for i in range(len(docs))]
        bow, tf = CountVectorizer(), TfidfVectorizer()
        out["bow"] = pd.DataFrame(bow.fit_transform(docs).toarray(), columns=bow.get_feature_names_out(),
                                  index=idx).reset_index(names="Document")
        out["tfidf"] = pd.DataFrame(tf.fit_transform(docs).toarray().round(3), columns=tf.get_feature_names_out(),
                                    index=idx).reset_index(names="Document")
    return out


@st.cache_data(show_spinner=False)
def train_w2v(docs):
    from gensim.models import Word2Vec  # slow import, so only when asked for

    model = Word2Vec([d.split() for d in docs], vector_size=50, window=5,
                     min_count=1, workers=1, seed=42, epochs=50)
    return {w: model.wv[w].tolist() for w in model.wv.index_to_key}


def build_journey(sentence, lex):
    R = res()
    cleaned = clean_text(sentence)
    return {
        "raw": sentence,
        "tokens": R.tokenize(sentence),
        "clean": cleaned,
        "lemmas": [lemma_of(lex, w)[0] for w in R.tokenize(cleaned) if w.isalpha() and w not in R.stop],
    }


def run_pipeline(text):
    res()  # shows its own one-time spinner on the very first run
    slot = st.empty()
    data = {}
    stages = [
        lambda: (s_spacy(text), s_tokenize(text))[1],  # parse once up front; later stages reuse it
        lambda: s_clean(text),
        lambda: s_normalize(text),
        lambda: s_tag(text),
        lambda: s_vectorize(tuple(data["sentences"])),
    ]
    for i, fn in enumerate(stages):
        slot.markdown(loader_html(i), unsafe_allow_html=True)
        data.update(fn())
    data["journey"] = build_journey(data["sentences"][0], s_spacy(text)["lex"]) if data["sentences"] else None
    slot.empty()
    return data


# ---------------------------------------------------------------- state
st.session_state.setdefault("text", "")
st.session_state.setdefault("analyzed", "")
st.session_state.setdefault("result", None)


def run():
    st.session_state.analyzed = st.session_state.text.strip()


def load_demo():
    st.session_state.text = DEMO_TEXT
    run()


def clear():
    st.session_state.text = ""
    st.session_state.analyzed = ""
    st.session_state.result = None


# ---------------------------------------------------------------- header + input
st.markdown("<h1 class='lab-title'>NLP Pipeline Lab</h1>", unsafe_allow_html=True)
st.markdown("<p class='lab-sub'>Paste some text and watch it move through tokenization, cleaning, "
            "normalization, tagging and vectorization.</p>", unsafe_allow_html=True)
st.markdown("<div class='dots'>" + "".join(f"<i style='background:{c}'></i>" for c in C.values()) + "</div>",
            unsafe_allow_html=True)
st.write("")

with st.container(border=True):
    st.text_area("Your text", key="text", height=120, label_visibility="collapsed",
                 placeholder="Paste or type text here, then choose Run pipeline.")
    b1, b2, b3, _ = st.columns([1.2, 1.3, 0.8, 5])
    b1.button("Run pipeline", type="primary", on_click=run, disabled=not st.session_state.text.strip())
    b2.button("Use demo text", on_click=load_demo)
    b3.button("Clear", on_click=clear)

text = st.session_state.analyzed
if not text:
    guide = [("tok", "Tokenization", "Compare three ways to split text into words."),
             ("clean", "Cleaning", "Strip emoji, URLs and noise. See what stop words go."),
             ("norm", "Normalization", "Reduce words to stems and lemmas."),
             ("ling", "Linguistics", "Tag parts of speech and find named entities."),
             ("vec", "Vectorization", "Turn sentences into numbers a model can use.")]
    st.markdown("<div class='empty'>" + "".join(
        f"<div style='--c:{C[k]}'><b>{t}</b><span>{d}</span></div>" for k, t, d in guide) + "</div>",
        unsafe_allow_html=True)
    st.caption(LIBS_NOTE)
    st.stop()

# run once per new text; reuse the stored result on every other rerun
if st.session_state.result is None or st.session_state.result["text"] != text:
    st.session_state.result = {"text": text, "data": run_pipeline(text)}
r = st.session_state.result["data"]

if st.session_state.text.strip() != text:
    st.markdown("<div class='note' style='--c:#e8920c'>The text has changed since the last run. "
                "Choose Run pipeline to update the results.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- tabs
tabs = st.tabs(["Overview", "Tokenization", "Cleaning", "Normalization",
                "Linguistics", "Vectorization", "Comparisons"])

with tabs[0]:
    st.markdown("<div class='stats'>" + stat("Characters", len(text), "#171a2b")
                + stat("Sentences", len(r["sentences"]), C["tok"])
                + stat("Tokens", len(r["tokens"]), C["clean"])
                + stat("After stop words", len(r["filtered"]), C["norm"])
                + stat("Entities", len(r["ents"]), C["ling"]) + "</div>", unsafe_allow_html=True)

    j = r["journey"]
    if j:
        st.subheader("One sentence, four stages")
        st.markdown(
            "<div>"
            f"<div class='row'><div class='lab' style='--c:#8890ad'><i></i>Raw</div><div class='val'>{html.escape(j['raw'])}</div></div>"
            f"<div class='row'><div class='lab' style='--c:{C['tok']}'><i></i>Tokenized</div><div class='val'>{chips(j['tokens'], C['tok'])}</div></div>"
            f"<div class='row'><div class='lab' style='--c:{C['clean']}'><i></i>Cleaned</div><div class='val'>{html.escape(j['clean'])}</div></div>"
            f"<div class='row'><div class='lab' style='--c:{C['norm']}'><i></i>Lemmas</div><div class='val'>{chips(j['lemmas'], C['norm'])}</div></div>"
            "</div>", unsafe_allow_html=True)

    st.subheader("In practice")
    use_case = st.selectbox("Pick a use case", list(USE_CASES), key="use_case")
    st.caption(USE_CASES[use_case][0])
    flow = ["User input", "Preprocessing", "Features", "Model", use_case]
    st.markdown("<div class='flow'>" + "<span class='arrow'>→</span>".join(
        f"<span class='step{' last' if i == len(flow) - 1 else ''}'>{html.escape(s)}</span>"
        for i, s in enumerate(flow)) + "</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='note' style='--c:{C['tok']}'>{html.escape(USE_CASES[use_case][1])}</div>",
                unsafe_allow_html=True)

with tabs[1]:
    st.subheader("Same text, three tokenizers")
    st.caption("Notice how each one handles contractions, punctuation, emoji and the email address.")
    c1, c2, c3 = st.columns(3)
    for col, title, note, key in [
        (c1, "Python split()", "Splits on whitespace only. Punctuation stays attached.", "split"),
        (c2, "NLTK word_tokenize", "Separates punctuation and splits contractions.", "nltk"),
        (c3, "spaCy", "Rule-based and language-aware. Keeps URLs and emails whole.", "spacy"),
    ]:
        with col:
            st.markdown(f"**{title}** · {len(r[key])} tokens")
            st.caption(note)
            st.markdown(chips(r[key], C["tok"]), unsafe_allow_html=True)

with tabs[2]:
    st.subheader("Cleaning")
    st.caption("Lowercase, remove emoji, URLs and emails, keep only letters and apostrophes.")
    st.markdown("**Before**")
    st.markdown(f"<div class='panel' style='--c:#8890ad'>{html.escape(text)}</div>", unsafe_allow_html=True)
    st.markdown("**After**")
    st.markdown(f"<div class='panel' style='--c:{C['clean']}'>{html.escape(r['cleaned']) or '(empty)'}</div>",
                unsafe_allow_html=True)
    st.markdown("**Stop words**")
    st.caption("Struck-through words are removed. The rest are kept.")
    stop_set = res().stop
    st.markdown("<div class='chips'>" + "".join(
        f"<span class='chip{' muted' if w in stop_set else ''}' style='--c:{C['clean']}'>{html.escape(w)}</span>"
        for w in r["tokens"]) + "</div>", unsafe_allow_html=True)

with tabs[3]:
    st.subheader("Normalization")
    st.caption("Stemmers chop suffixes by rule. Lemmatizers look words up, so the output is a real word. "
               "Each unique word appears once, and lemmatizers use its part of speech.")
    if r["norm"].empty:
        st.warning("Not enough words to normalize.")
    else:
        n = r["norm"]
        st.markdown("<div class='stats' style='grid-template-columns:repeat(4,1fr)'>"
                    + stat("Unique words", n["Original"].nunique(), "#171a2b")
                    + stat("After Porter", n["Porter"].nunique(), C["norm"])
                    + stat("After Lancaster", n["Lancaster"].nunique(), C["norm"])
                    + stat("After lemmatizing", n["NLTK lemma"].nunique(), C["norm"]) + "</div>",
                    unsafe_allow_html=True)
        show_df(n)

with tabs[4]:
    st.subheader("Part-of-speech tags")
    left, right = st.columns([2, 1])
    with left:
        show_df(r["pos"])
    with right:
        st.caption("Tag distribution")
        st.bar_chart(r["pos"]["POS"].value_counts(), color=C["ling"])
    st.subheader("Named entities")
    if r["ents"]:
        st.markdown("<div class='chips'>" + "".join(
            f"<span class='chip' style='--c:{C['ling']}' title='{html.escape(d)}'>{html.escape(t)}<b>{html.escape(l)}</b></span>"
            for t, l, d in r["ents"]) + "</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='note' style='--c:{C['ling']}'>spaCy found no named entities in this text.</div>",
                    unsafe_allow_html=True)

with tabs[5]:
    st.subheader("Vectorization")
    docs = r["docs"]
    if not docs:
        st.warning("Not enough meaningful words to vectorize.")
    else:
        st.caption(f"Each sentence is treated as a document ({len(docs)} total), so TF-IDF has something to compare.")
        left, right = st.columns(2)
        with left:
            st.markdown("**Bag of Words**")
            show_df(r["bow"])
        with right:
            st.markdown("**TF-IDF**")
            show_df(r["tfidf"])

        st.markdown("**Word2Vec embedding**")
        if st.checkbox("Train Word2Vec on these sentences", key="w2v"):
            with st.spinner("Training Word2Vec on your sentences…"):
                vectors = train_w2v(tuple(docs))
            word = st.selectbox("Word", list(vectors), label_visibility="collapsed")
            st.bar_chart(pd.Series(vectors[word], name=word), color=C["vec"])
            st.caption("Trained on your text only, so these 50 numbers are for illustration. "
                       "Real embeddings need millions of words.")
        else:
            st.caption("Off by default because gensim is slow to load. Tick the box to train a small model.")

with tabs[6]:
    st.subheader("Tokenizer comparison")
    show_df(pd.DataFrame({
        "Method": ["Python split()", "NLTK", "spaCy"],
        "Language aware": ["No", "Partly", "Yes"],
        "Handles punctuation": ["No", "Yes", "Yes"],
        "Production ready": ["No", "Partly", "Yes"],
    }))
    st.subheader("Stemming vs lemmatization")
    show_df(pd.DataFrame({
        "Aspect": ["Accuracy", "Speed", "Output is a real word"],
        "Stemming": ["Lower", "Faster", "Not always"],
        "Lemmatization": ["Higher", "Slower", "Yes"],
    }))
    st.caption(LIBS_NOTE)