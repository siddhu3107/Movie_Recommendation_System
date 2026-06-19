import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies.csv")

# Keep useful columns
df = df[['title', 'genres', 'overview']]

# Remove missing values
df.dropna(inplace=True)

# Combine text data
df['tags'] = df['genres'] + df['overview']

# Convert text into numbers
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(df['tags']).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):

    movie = movie.lower()

    # Find movie index
    movie_index = df[df['title'].str.lower() == movie].index

    if len(movie_index) == 0:
        print("Movie not found")
        return

    movie_index = movie_index[0]

    # Get similarity scores
    distances = similarity[movie_index]

    # Sort movies
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print("\nRecommended Movies:\n")

    for i in movies_list:
        print(df.iloc[i[0]].title)

# User input
movie_name = input("Enter movie name: ")

recommend(movie_name)