import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model for generating query embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_faiss_index(index_path="playstore_reviews.index", data_path="faiss-index-df.csv"):

    # Load the FAISS index from the file
    index = faiss.read_index(index_path)
    st.write(f"FAISS index loaded from {index_path}")
    
    # Load the data into a Pandas DataFrame
    data = pd.read_csv(data_path)
    st.write(f"Data loaded from {data_path}, containing {len(data)} records.")
    
    return index, data

index, data = load_faiss_index(index_path="playstore_reviews.index", data_path="faiss-index-df.csv")

# Load FAISS index (assuming the index is already saved as 'faiss_index.bin')
# index = faiss.read_index('.\faiss-index-df.index')

# Load reviews data (assuming you have a file with the reviews stored)
# Example: load the reviews stored in a file (reviews.npy)
# reviews = pd.read_csv('faiss-index-df.csv')
# reviews.rename(columns={"Unnamed: 0":"Index"},inplace=True)

# reviews.set_index("Index",inplace=True)
# df=df[["content"]]



st.title("Text Retrieval Using FAISS")
st.write("This app allows you to retrieve the most relevant reviews based on your query.")

# Number of results to return
top_k = st.slider("Select number of similar reviews to retrieve:", min_value=1, max_value=20, value=5)

# Input query from the user
query = st.text_input("Enter your query to find similar reviews:")
show_scores = st.checkbox("Show similarity scores")
# When the user clicks the search button
if st.button("Search"):
    if query:
        # Generate query embeddings using SBERT
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        D, I = index.search(np.array(query_embedding, dtype=np.float32), top_k)

        # Display similar reviews
        st.subheader("Top Similar Reviews:")
        for i in range(top_k):
            review_index = I[0][i]  # Get the index of the retrieved review from FAISS
            review_text = data.iloc[review_index]['content'] 
            st.write(f"**Review {i+1}:** {review_text}")
            if show_scores:
                st.write(f"**Similarity score:** {D[0][i]:.4f}")
    else:
        st.write("Please enter a valid query.")
    # Optionally display similarity scores
    
    
        st.write("Similarity Scores:", D)