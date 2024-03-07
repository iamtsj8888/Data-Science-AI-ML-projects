#!/usr/bin/env python
# coding: utf-8

# In[114]:


import pandas as pd
import numpy as np


# In[115]:


credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')


# In[116]:


movies.head()


# In[117]:


credits.head()


# In[118]:


movies = movies.merge(credits, on='title')


# In[119]:


movies.head()


# In[120]:


movies.shape


# In[121]:


movies.columns


# In[122]:


movies.head()


# In[123]:


movies.dropna(inplace=True)


# In[124]:


movies.isnull().sum()


# In[125]:


movies.iloc[0]['genres']


# In[126]:


import ast
def convert(text):
    h = []
    for i in ast.literal_eval(text):
        h.append(i['name'])
    return h


# In[127]:


movies['genres'] = movies['genres'].apply(convert)


# In[128]:


movies.head()


# In[129]:


movies.iloc[0]['keywords']


# In[130]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[131]:


movies.iloc[0]['cast']


# In[132]:


movies.iloc[0]['crew']


# In[133]:


def pull_director(text):
    D = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            D.append(i['name'])
            break
        return D


# In[134]:


movies['crew'] = movies['crew'].apply(pull_director)
movies.head()


# In[135]:


movies.iloc[0]['overview']


# In[136]:


movies['overview'] = movies['overview'].apply(lambda 
x:x.split())
movies.head()


# In[137]:


movies.iloc[0]['overview']


# In[138]:


def remove_space(L):
    S = []
    for i in L:
        S.append(i.replace(" ",""))
    return S


# In[139]:


movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew']
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)


# In[140]:


movies.head()


# In[141]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[142]:


movies.head()


# In[143]:


movies.iloc[0]['tags']


# In[144]:


Movie_df = movies[['movie_id','title','tags']]


# In[145]:


Movie_df.head()


# In[146]:


Movie_df.iloc[0]['tags']


# In[154]:


Movie_df['tags'] = Movie_df['tags'].apply(lambda x:x)


# In[155]:


print(Movie_df.head)


# In[161]:


Movie_df['tags'] = Movie_df['tags'].astype('str')


# In[162]:


Movie_df['tags'].fillna('', inplace=True)


# In[163]:


print(Movie_df['tags'].head())


# In[164]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[165]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(Movie_df['tags'].values.astype('U'))  


# In[166]:


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[167]:


def get_recommendations(title, cosine_sim=cosine_sim):
    
    idx = Movie_df[Movie_df['title'] == title].index[0]

   
    sim_scores = list(enumerate(cosine_sim[idx]))

   
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)


    sim_scores = sim_scores[1:11]


    movie_indices = [i[0] for i in sim_scores]

    
    return Movie_df['title'].iloc[movie_indices]


# In[168]:


film_basligi = "Interstellar" 
print("Film Önerileri:")
print(get_recommendations(film_basligi))


# In[ ]:




