# AI Job Matching System
# Complete code for local execution

import pandas as pd
import re
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

print("=" * 50)
print("AI JOB MATCHING SYSTEM")
print("=" * 50)

# Step 1: Load the data
print("\n1. LOADING DATASET...")
df = pd.read_csv('Resume.csv')
print(f"   Loaded {len(df)} resumes")
print(f"   Columns: {list(df.columns)}")

# Step 2: Load spaCy
print("\n2. LOADING LANGUAGE MODEL...")
nlp = spacy.load('en_core_web_sm')
print("   ✓ Language model loaded")

# Step 3: Define cleaning functions
print("\n3. CREATING TEXT PROCESSING FUNCTIONS...")

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_text(text):
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(words)

print("   ✓ Functions created")

# Step 4: Process all resumes
print("\n4. PROCESSING ALL RESUMES (this will take 2-3 minutes)...")
df['cleaned'] = df['Resume'].apply(clean_text)
print("   ✓ Cleaning complete")
df['processed'] = df['cleaned'].apply(lemmatize_text)
print("   ✓ Lemmatization complete")

# Step 5: Split data
print("\n5. SPLITTING DATA FOR TRAINING/TESTING...")
X = df['processed']
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Training: {len(X_train)} resumes")
print(f"   Testing: {len(X_test)} resumes")

# Step 6: Vectorize text
print("\n6. CONVERTING TEXT TO NUMBERS (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
print(f"   Created {X_train_vectorized.shape[1]} word features")

# Step 7: Train model
print("\n7. TRAINING CLASSIFICATION MODEL...")
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)
print("   ✓ Training complete")

# Step 8: Evaluate
print("\n8. EVALUATING MODEL PERFORMANCE...")
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"   Accuracy: {accuracy:.2%}")
print(f"   Correctly predicted {accuracy*len(y_test):.0f} out of {len(y_test)} resumes")

# Step 9: Create similarity matrix
print("\n9. BUILDING SIMILARITY MATCHER...")
all_vectorized = vectorizer.transform(df['processed'])
similarity_matrix = cosine_similarity(all_vectorized)
print(f"   Created similarity matrix: {similarity_matrix.shape[0]} × {similarity_matrix.shape[1]}")

# Step 10: Create matching function
def find_similar_resumes(resume_index, top_n=5):
    scores = similarity_matrix[resume_index]
    similar_indices = np.argsort(scores)[-top_n-1:-1][::-1]
    
    print(f"\n   Original: Resume #{resume_index} - {df.iloc[resume_index]['Category']}")
    print(f"   TOP {top_n} MATCHES:")
    for idx in similar_indices:
        score = scores[idx]
        print(f"   → Resume #{idx} | Match: {score:.1%} | {df.iloc[idx]['Category']}")

def match_new_resume(new_text):
    cleaned = clean_text(new_text)
    processed = lemmatize_text(cleaned)
    new_vector = vectorizer.transform([processed])
    similarities = cosine_similarity(new_vector, all_vectorized).flatten()
    top_indices = np.argsort(similarities)[-5:][::-1]
    
    print("\n   TOP 5 MATCHES FOR YOUR RESUME:")
    for idx in top_indices:
        score = similarities[idx]
        print(f"   → Match: {score:.1%} | {df.iloc[idx]['Category']}")

# Step 11: Show results
print("\n" + "=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
print(f"Total Resumes: {len(df)}")
print(f"Job Categories: {df['Category'].nunique()}")
print(f"Model Accuracy: {accuracy:.2%}")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")
print(f"Features Used: {X_train_vectorized.shape[1]}")

# Step 12: Demonstrate matching
print("\n" + "=" * 50)
print("DEMONSTRATION")
print("=" * 50)
find_similar_resumes(0, top_n=3)

print("\n" + "=" * 50)
print("TEST WITH YOUR OWN TEXT")
print("=" * 50)
my_text = """
I am a software developer with Python experience
and machine learning knowledge.
"""
match_new_resume(my_text)

print("\n" + "=" * 50)
print("PROGRAM COMPLETE")
print("=" * 50)