import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies.csv")

# Keep useful columns
df = df[['title', 'genres', 'overview']]

# Remove missing values
df.dropna(inplace=True)

# Combine columns
df['tags'] = df['genres'] + df['overview']

# Convert text to vectors
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(df['tags']).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):

    movie = movie.lower()

    movie_index = df[df['title'].str.lower() == movie].index

    if len(movie_index) == 0:
        return ["Movie not found"]

    movie_index = movie_index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i in movies_list:
        recommendations.append(df.iloc[i[0]].title)

    return recommendations

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(page_title="Movie Recommendation System")

st.title("🎬 Movie Recommendation System")

st.write("Get movie recommendations using Machine Learning")

movie_name = st.text_input("Enter movie name")

if st.button("Recommend"):

    if movie_name.strip() == "":
        st.warning("Please enter a movie name")

    else:

        results = recommend(movie_name)

        st.subheader("Recommended Movies:")

        for movie in results:
            st.write("✅", movie)