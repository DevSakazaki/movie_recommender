import pandas as pd
import json 
import streamlit as st 
from time import sleep
from stqdm import stqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="More Films",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# App feito por Sakazaki"
    }
)

st.title("Descubra novos filmes!")
user_input2 = st.slider('Quantos filmes você deseja?', 2, 20)
user_input = st.text_input("Me fale um filme... (Em inglês!!) ")
if user_input == ' ':
  st.write("Ué... Você ainda não me falou nenhum filme!")

df = pd.read_csv('tmdb_5000_movies.csv')
x = df.iloc[0]
j = json.loads(x['genres'])

def genres_and_keywords_to_string(row): 
  genres = json.loads(row['genres'])
  genres = ' '.join(''.join(j['name'].split()) for j in genres)

  keywords = json.loads(row['keywords'])
  keywords = ' '.join(''.join(j['name'].split()) for j in keywords)
  return "%s %s" % (genres, keywords) 
df['string'] = df.apply(genres_and_keywords_to_string, axis=1) 


tfidf = TfidfVectorizer(max_features=2000)
X = tfidf.fit_transform(df['string'])
movie2idx = pd.Series(df.index, index=df['title'])

def recommend(user_input): 
  idx = movie2idx[user_input]
  if type(idx) == pd.Series:
    idx = idx.iloc[0]

  query = X[idx]
  scores = cosine_similarity(query, X)
  scores = scores.flatten()

  recommended_idx = (-scores).argsort()[1:user_input2]

  st.subheader("Recomendações para o filme " + user_input)

  recommended_movies = df['title'].iloc[recommended_idx].tolist()
  recommended_movies_text = '\n'.join([movie + '\n' for movie in recommended_movies])

  return recommended_movies_text

if user_input:
    for _ in stqdm(range(user_input2), desc="Estou procurandos os melhores filmes para você... ", mininterval=1):
        sleep(0.5)
    st.write(recommend(user_input))
