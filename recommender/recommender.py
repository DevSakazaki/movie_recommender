import pandas as pd
import json 
import streamlit as st 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="More Films",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Report a bug': "https://github.com/DevSakazaki",
        'About': 'App de recomendação de filmes feito por SergioJr, use nome de filmes em inglês, filmes mais recentes não serão reconhecidos. Nossa base de dados atualmente se limita somente a apenas 5000 filmes, porém aumentaremos em breves atualizações, fique atento! Acesse meu Portfólio: https://portfolio-jet-chi-78.vercel.app/index.html'
    }
)

st.title("Descubra novos filmes!")
quantidade = st.slider('Quantos filmes você deseja?', 2, 20)
user_input = st.text_input("Me fale um filme... (Em inglês!!) ")
if user_input == ' ':
  st.write("Ué... Você ainda não me falou nenhum filme!")

df = pd.read_csv('recommender/data/tmdb_5000_movies.csv')
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
  if not df['title'].str.contains(user_input).any():
        st.subheader('Ainda não conheço o filme ' + user_input + ', desculpe :(')
        st.write('Que tal tentar um filme um pouco mais conhecido?')
        return None
  
  idx = movie2idx[user_input]
  if type(idx) == pd.Series:
    idx = idx.iloc[0]

  query = X[idx]
  scores = cosine_similarity(query, X)
  scores = scores.flatten()

  recommended_idx = (-scores).argsort()[1:quantidade]

  st.subheader("Recomendações para o filme " + user_input + ", espero que goste!")

  recommended_movies = df['title'].iloc[recommended_idx].tolist()
  recommended_movies_text = '\n'.join([movie + '\n' for movie in recommended_movies])

  return recommended_movies_text

if user_input:
    st.write(recommend(user_input))
