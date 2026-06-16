import pandas as pd
import numpy as np

movies = pd.read_csv('movies.csv')

movies = movies[['id','title','overview','release_date','popularity','vote_average','vote_count']]

movies.dropna(inplace=True)

movies['year'] = movies['release_date'].apply(lambda x: str(x).split('/')[-1])

movies['tags'] = movies['overview'] + " " + movies['year'] + " " + movies['title']

movies['tags'] = movies['tags'].apply(lambda x: x.lower())

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie = movie.lower()
    movies['title_lower'] = movies['title'].str.lower()

    if movie not in movies['title_lower'].values:
        print("Movie not found!")
        return

    index = movies[movies['title_lower'] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    print("\nRecommended Movies:\n")
    for i in movie_list:
        print(movies.iloc[i[0]].title)
        
recommend('The Godfather')