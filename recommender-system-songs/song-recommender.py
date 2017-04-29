
# coding: utf-8

# ## Building a song recommender
# 
# 
# ### Fire up GraphLab Create

# In[1]:

import graphlab


# ### Load music data

# In[2]:

song_data = graphlab.SFrame('song_data.gl/')


# ### Explore data
# 
# Music data shows how many times a user listened to a song, as well as the details of the song.

# In[3]:

song_data.head()


# ### Most popular songs in the dataset

# In[4]:

graphlab.canvas.set_target('ipynb')


# In[5]:

song_data['song'].show()


# In[6]:

len(song_data)


# ### Unique users in the dataset

# In[7]:

users = song_data['user_id'].unique()


# In[8]:

len(users)


# ### Build a song recommender

# In[9]:

train_data,test_data = song_data.random_split(.8,seed=0)


# ### Simple popularity-based recommender

# In[10]:

popularity_model = graphlab.popularity_recommender.create(train_data,
                                                         user_id='user_id',
                                                         item_id='song')


# ### Apply the popularity model to make some predictions
# 
# A popularity model makes the same prediction for all users, so provides no personalization.

# In[11]:

popularity_model.recommend(users=[users[0]])


# In[12]:

popularity_model.recommend(users=[users[1]])


# ### Build a song recommender with personalization
# 
# We now create a model that allows us to make personalized recommendations to each user. 

# In[13]:

personalized_model = graphlab.item_similarity_recommender.create(train_data,
                                                                user_id='user_id',
                                                                item_id='song')


# ### Applying the personalized model to make song recommendations
# 
# As you can see, different users get different recommendations now.

# In[14]:

personalized_model.recommend(users=[users[0]])


# In[15]:

personalized_model.recommend(users=[users[1]])


# ### Apply the model to find similar songs to any song in the dataset

# In[16]:

personalized_model.get_similar_items(['With Or Without You - U2'])


# In[17]:

personalized_model.get_similar_items(['Chan Chan (Live) - Buena Vista Social Club'])


# ### Quantitative comparison between the models
# 
# We now formally compare the popularity and the personalized models using precision-recall curves. 

# In[20]:

model_performance = graphlab.compare(test_data, [popularity_model, personalized_model], user_sample=0.05)
graphlab.show_comparison(model_performance,[popularity_model, personalized_model])


# ### The curve shows that the personalized model provides much better performance. 

# ### Unique users who listened to Kanye West's songs

# In[23]:

kanye_songs = song_data[song_data['artist'] == 'Kanye West']


# In[24]:

kanye_songs


# In[30]:

kanye_users = kanye_songs['user_id'].unique()


# In[31]:

len(kanye_users)


# ### Unique users who listened to Foo Fighters songs

# In[32]:

foo_songs = song_data[song_data['artist'] == 'Foo Fighters']


# In[33]:

foo_songs


# In[34]:

foo_users = foo_songs['user_id'].unique()


# In[35]:

len(foo_users)


# ### Unique users who listened to Taylor Swift's songs

# In[36]:

taylor_songs = song_data[song_data['artist'] == 'Taylor Swift']


# In[37]:

taylor_songs


# In[38]:

taylor_users = taylor_songs['user_id'].unique()


# In[39]:

len(taylor_users)


# ### Unique users who listened to Lady GaGa's songs

# In[41]:

gaga_songs = song_data[song_data['artist'] == 'Lady GaGa']


# In[42]:

gaga_songs


# In[43]:

gaga_users = gaga_songs['user_id'].unique()


# In[44]:

len(gaga_users)


# ### Taylor swift has the highest unique users listening to her songs

# ### Total number of times songs played of each artist

# In[46]:

total_count_by_artist = song_data.groupby(key_columns='artist', operations={'total_count': graphlab.aggregate.SUM('listen_count')})


# In[50]:

sort_count = total_count_by_artist.sort('total_count', ascending=False)


# ### Top 10 most popular artists by song listen count

# In[51]:

sort_count


# ### Bottom 10 least popular artists by song listen count

# In[52]:

sort_count.tail()


# ### Most recommended song

# In[56]:

#Taking 10000 unique users
subset_test_users = test_data['user_id'].unique()[1:10000]


# In[58]:

#Recommending one song for each user
recommendations = personalized_model.recommend(subset_test_users, k=1)


# In[59]:

recommendations


# In[65]:

most_recommended_song = recommendations.groupby(key_columns='song', operations={'total_count': graphlab.aggregate.COUNT()})


# In[66]:

most_recommended_song_sorted = most_recommended_song.sort('total_count', ascending=False)


# In[67]:

most_recommended_song_sorted


# ### So Undo by Bjork is the most recommended song from the subset data of 10000 unique users
