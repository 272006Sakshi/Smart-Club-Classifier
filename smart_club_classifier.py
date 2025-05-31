import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("📚 Smart Club Event Classifier & Recommender")

st.sidebar.header("🎯 Your Interests")
interests = st.sidebar.multiselect(
    "Choose your interests:",
    ["Tech", "Cultural", "Sports", "Poetry", "Robotics"]
)

uploaded_file = st.file_uploader("📁 Upload the file of events(in CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Event Data")
    st.dataframe(df.head())

    if "Category" in df.columns:
        # Step 1: Train classifier
        X = df['Description']
        y = df['Category']

        vectorizer = TfidfVectorizer()
        X_vectorized = vectorizer.fit_transform(X)

        model = MultinomialNB()
        model.fit(X_vectorized, y)

        st.success("✅ Model trained successfully!")

        # Step 2: Predict categories
        df['Predicted Category'] = model.predict(X_vectorized)

        st.subheader("🔍 Classified Events")
        st.dataframe(df[['Event Name', 'Description', 'Predicted Category']])

        # Step 3: Recommend based on user interests
        if interests:
            # Normalize capitalization (title case)
            interests_title = [i.title() for i in interests]
            recommended = df[df['Predicted Category'].isin(interests_title)]

            st.subheader("✨ Recommended Events For You")
            if not recommended.empty:
                st.dataframe(recommended[['Event Name', 'Description', 'Predicted Category']])
            else:
                st.info("No events match your interests. Try selecting different interests.")
        else:
            st.info("Please select at least one interest from the sidebar to get recommendations.")
    else:
        st.warning("🚫 Please include a 'Category' column in your CSV to train the model.")
        