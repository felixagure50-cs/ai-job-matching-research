import streamlit as st
import pandas as pd
import re
import subprocess
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page setup
st.set_page_config(page_title="AI Job Matching", layout="wide")
st.title("🔍 AI Driven Job Matching System")
st.write("Find matching jobs or candidates using AI")

# Function to download spaCy model if not available
@st.cache_resource
def load_spacy_model():
    try:
        import spacy
        # Try to load the model
        nlp = spacy.load("en_core_web_sm")
        st.success("✅ spaCy model loaded successfully!")
        return nlp
    except:
        st.info("Downloading spaCy language model (this happens once)...")
        # Download the model
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        import spacy
        nlp = spacy.load("en_core_web_sm")
        st.success("✅ spaCy model downloaded and loaded!")
        return nlp

# Load spaCy model
try:
    nlp = load_spacy_model()
except Exception as e:
    st.error(f"Error loading spaCy: {e}")
    st.stop()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Resume.csv')
    return df

@st.cache_data
def process_data(df, _nlp):
    def clean_text(text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def lemmatize_text(text):
        doc = _nlp(text)
        words = [token.lemma_ for token in doc if not token.is_stop]
        return ' '.join(words)
    
    progress_bar = st.progress(0)
    st.write("Processing resumes...")
    
    df['cleaned'] = df['Resume_str'].apply(clean_text)
    progress_bar.progress(30)
    
    df['processed'] = df['cleaned'].apply(lemmatize_text)
    progress_bar.progress(60)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    vectors = vectorizer.fit_transform(df['processed'])
    progress_bar.progress(80)
    
    matrix = cosine_similarity(vectors)
    progress_bar.progress(100)
    
    st.success("✅ Processing complete!")
    return df, vectorizer, vectors, matrix

# Load everything
with st.spinner("Loading system..."):
    df = load_data()
    st.write(f"✅ Dataset loaded: {len(df)} resumes")
    df, vectorizer, vectors, matrix = process_data(df, nlp)

st.success("✅ System ready!")

# Sidebar info
with st.sidebar:
    st.header("📊 Dataset Info")
    st.write(f"Total Resumes: {len(df)}")
    st.write(f"Job Categories: {df['Category'].nunique()}")
    st.write("\n📌 How it works:")
    st.write("1. Text is cleaned and processed")
    st.write("2. Converted to numbers using TF-IDF")
    st.write("3. Similarity calculated using cosine")

# Main app tabs
tab1, tab2, tab3 = st.tabs(["🎯 Find Similar Resumes", "📝 Match New Resume", "📊 Explore Categories"])

# Tab 1: Find similar resumes
with tab1:
    st.header("Find Resumes Similar to an Existing One")
    
    resume_index = st.number_input("Enter resume index (0 to " + str(len(df)-1) + "):", 
                                   min_value=0, max_value=len(df)-1, value=0)
    
    if st.button("Find Similar", key="btn1"):
        scores = matrix[resume_index]
        similar = np.argsort(scores)[-6:-1][::-1]
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Selected Resume")
            st.write(f"**Category:** {df.iloc[resume_index]['Category']}")
            st.write("**Preview:**")
            st.write(df.iloc[resume_index]['processed'][:300] + "...")
        
        with col2:
            st.subheader("Top 5 Similar Resumes")
            for idx in similar:
                score = scores[idx]
                st.write(f"**Match: {score:.1%}**")
                st.write(f"Category: {df.iloc[idx]['Category']}")
                st.write(f"Preview: {df.iloc[idx]['processed'][:150]}...")
                st.divider()

# Tab 2: Match new resume
with tab2:
    st.header("Match Your Own Resume")
    
    user_text = st.text_area("Paste your resume text here:", height=200)
    
    if st.button("Find Matches", key="btn2") and user_text:
        with st.spinner("Processing..."):
            cleaned = re.sub(r'[^a-zA-Z\s]', '', user_text.lower())
            doc = nlp(cleaned)
            processed = ' '.join([token.lemma_ for token in doc if not token.is_stop])
            
            user_vector = vectorizer.transform([processed])
            similarities = cosine_similarity(user_vector, vectors).flatten()
            top_indices = np.argsort(similarities)[-10:][::-1]
            
            st.subheader("Top 10 Matching Resumes")
            cols = st.columns(2)
            for i, idx in enumerate(top_indices):
                with cols[i % 2]:
                    score = similarities[idx]
                    st.write(f"**Match: {score:.1%}**")
                    st.write(f"Category: {df.iloc[idx]['Category']}")
                    st.write(f"Preview: {df.iloc[idx]['processed'][:200]}...")
                    st.divider()

# Tab 3: Explore categories
with tab3:
    st.header("Explore Job Categories")
    
    category_counts = df['Category'].value_counts()
    st.bar_chart(category_counts)
    
    selected_category = st.selectbox("Select a category:", df['Category'].unique())
    
    samples = df[df['Category'] == selected_category].head(3)
    for i, row in samples.iterrows():
        with st.expander(f"Sample {i}"):
            st.write(row['processed'][:500] + "...")

st.markdown("---")
st.markdown("Built with Python, scikit-learn, spaCy, and Streamlit")
