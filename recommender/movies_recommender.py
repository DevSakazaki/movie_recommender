import pandas as pd 
import matplotlib.pyplot as plt
import json 
import streamlit as st 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.write("Descubra novos filmes!")

user_input = st.text_input("Me fale um filme... ")

df = pd.read_csv('tmdb_5000_movies.csv')
df.head()
x = df.iloc[0]
x

j = json.loads(x['genres'])
j

def genres_and_keywords_to_string(row): 
  genres = json.loads(row['genres'])
  genres = ' '.join(''.join(j['name'].split()) for j in genres)

  keywords = json.loads(row['keywords'])
  keywords = ' '.join(''.join(j['name'].split()) for j in keywords)
  return "%s %s" % (genres, keywords) 

df['string'] = df.apply(genres_and_keywords_to_string, axis=1) 


tfidf = TfidfVectorizer(max_features=2000)

X = tfidf.fit_transform(df['string'])

X

movie2idx = pd.Series(df.index, index=df['title'])
movie2idx

idx = movie2idx['Love Actually']
idx

query = X[idx]
query

query.toarray()
#Matriz query

#calcular a similaridade de cosseno entre nosso vetor query e todos os vetores em X 
#Observe que isso inclui o próprio vetor query, pois ele venho de X
scores = cosine_similarity(query, X)
scores  

#A maioria dos valores são 0 pois se dois filmes não compartilham termos em comum, seu produto será 0, logo a similaridade também será 0

#Atualmente o array tem o formato de 1xN, e aqui achatamos ela para um array 1-D
scores = scores.flatten()

plt.plot(scores)

(-scores).argsort()

plt.plot(scores[(-scores).argsort()])
recommended_idx = (-scores).argsort()[1:6]

#Convertendo indices de volta para titulos 
df['title'].iloc[recommended_idx]

#Veja, para o filme "Scream 3" chegamos ao resultado de Sexta feira 13 etc, todos filmes de terror

"""# Interface"""

if user_input == ' ':
  print("Poxa... Você não me falou nenhum filme!")

"""# Função de recomendação"""

def recommend(user_input): 
  idx = movie2idx[user_input]
  if type(idx) == pd.Series:
    idx = idx.iloc[0]

  query = X[idx]
  scores = cosine_similarity(query, X)
  scores = scores.flatten()

  recommended_idx = (-scores).argsort()[1:6]

  return df['title'].iloc[recommended_idx]

st.write("Recomendações para o filme",user_input,", espero que curta:")
st.write(recommend(user_input))