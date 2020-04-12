#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#import of review data
cols = ["user id","item id","rating","timestamp"]
#encoding using ISO-8859-1 is used because utf-8 does not support all the characters in movie names
df_data = pd.read_csv("ml-100k/u.data",sep="\t",names=cols,header=None,encoding="ISO-8859-1")


# In[3]:


#import of moviedata
cols = ["movie id",
        "movie title",
        "release date",
        "video release date",
        "IMDb URL","unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western"]

df_movie = pd.read_csv("ml-100k/u.item",sep="|",names=cols,header=None,encoding="ISO-8859-1")


# In[5]:


#import of user data
cols = ["user id","age","gender","occupation","zip code"]
df_user = pd.read_csv("ml-100k/u.user",sep="|",names=cols,header=None,encoding="ISO-8859-1")


# In[6]:


#frequency binning the ages into age groups as it will be easier for future analysis
df_user['age_group'] = pd.qcut(df_user['age'],q=10,precision=0)

#the bins are of unequal size due to repeating values in a bin
df_user['age_group'].value_counts()


# In[7]:


df_movie.drop(["movie id",
               "movie title",
               "release date",
               "video release date",
               "IMDb URL",
               "unknown"],axis=1).sum(axis = 0, skipna = True)


# In[8]:


df = pd.merge(pd.merge(df_data,
                  df_user[["user id",
                           "age",
                           "gender",
                           "occupation"]],
                  on='user id',
                  how='left'),
              df_movie,
              left_on = 'item id',
              right_on = 'movie id',
              how ='left')


# In[9]:


df_genre = df[["rating",
                "Action",
                "Adventure",
               "Animation",
               "Children's",
                "Comedy",
                "Crime",
                "Documentary",
                "Drama",
                "Fantasy",
                "Film-Noir",
                "Horror",
                "Musical",
                "Mystery",
                "Romance",
                "Sci-Fi",
                "Thriller",
                "War",
                "Western"]]


# In[10]:


def select_genre(row):
    for key,value in row.items():
        if value==1:
            return key


# In[13]:


df_genre['genre']= df_genre.apply(lambda row: select_genre(row.iloc[2:]),axis=1)
df_genre.drop(["Action",
                   "Adventure",
                   "Animation",
                   "Children's",
                   "Comedy",
                   "Crime",
                   "Documentary",
                   "Drama",
                   "Fantasy",
                   "Film-Noir",
                   "Horror",
                   "Musical",
                   "Mystery",
                   "Romance",
                   "Sci-Fi",
                   "Thriller",
                   "War",
                   "Western"],
                 inplace=True,
                 axis=1)


# In[16]:


df_genre_grouped = df_genre.groupby(['genre']).mean()

df_dict = dict{}

for row in df_genre_grouped:
# In[ ]:




