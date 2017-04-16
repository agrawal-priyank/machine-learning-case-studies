
# coding: utf-8

# # Document retrieval from wikipedia data
# 
# ### Fire up GraphLab Create

# In[1]:

import graphlab


# ### Load data consisting about information of famous people from wikipedia

# In[2]:

people = graphlab.SFrame('people_wiki.gl/')


# In[3]:

people.head()


# ### Explore the dataset
# 
# ### Exploring information on Barack Obama

# In[5]:

obama = people[people['name'] == 'Barack Obama']


# In[6]:

obama


# ### Exploring the entry for actor George Clooney

# In[8]:

clooney = people[people['name'] == 'George Clooney']
clooney['text']


# ### Get the word counts for Obama article

# In[9]:

obama['word_count'] = graphlab.text_analytics.count_words(obama['text'])


# In[10]:

print obama['word_count']


# ### Turning dictonary of word counts into a table

# In[12]:

obama_word_count_table = obama[['word_count']].stack('word_count', new_column_name = ['word','count'])


# In[13]:

obama_word_count_table.head()


# ### Sort the word counts for the Obama article

# In[18]:

obama_word_count_table.sort('count', ascending=False)


# ### Most common words include uninformative words like "the", "in", "and", etc.

# ### Compute TF-IDF for the corpus 
# 
# To give more weight to informative words, we weigh them by their TF-IDF scores.

# In[14]:

people['word_count'] = graphlab.text_analytics.count_words(people['text'])
people.head()


# In[21]:

people['tfidf'] = graphlab.text_analytics.tf_idf(people['word_count'])


# ### Examine the TF-IDF for the Obama article

# In[23]:

obama = people[people['name'] == 'Barack Obama']


# In[24]:

obama[['tfidf']].stack('tfidf', new_column_name=['word', 'tfidf']).sort('tfidf', ascending=False)


# ### Words with highest TF-IDF are much more informative.

# ### Compute distances between a few people

# In[25]:

clinton = people[people['name'] == 'Bill Clinton']


# In[26]:

beckham = people[people['name'] == 'David Beckham']


# ### Is Obama closer to Clinton?

# In[27]:

graphlab.distances.cosine(obama['tfidf'][0], clinton['tfidf'][0])


# ### Is Obama closer to Beckham than Clinton?

# In[28]:

graphlab.distances.cosine(obama['tfidf'][0], beckham['tfidf'][0])


# ### Build a nearest neighbor model for document retrieval

# In[29]:

knn_model = graphlab.nearest_neighbors.create(people, features=['tfidf'], label='name')


# ### Applying the model

# ### Top 5 people closest to Obama

# In[30]:

knn_model.query(obama)


# ### Top 5 people closest to Taylor Swift

# In[31]:

swift = people[people['name'] == 'Taylor Swift']


# In[32]:

knn_model.query(swift)


# ### Top 5 people closest to Angelina Jolie

# In[33]:

jolie = people[people['name'] == 'Angelina Jolie']


# In[34]:

knn_model.query(jolie)


# ### Top 5 people closest to Arnold

# In[36]:

arnold = people[people['name'] == 'Arnold Schwarzenegger']


# In[37]:

knn_model.query(arnold)


# ### Exploring article on Elton John

# In[43]:

john = people[people['name'] == 'Elton John']


# In[44]:

john


# ### Create a table and sort the word counts of Elton John article

# In[47]:

john_word_count_table = john[['word_count']].stack('word_count', new_column_name = ['word', 'count'])


# In[51]:

john_word_count_sorted = john_word_count_table.sort('count', ascending = False)


# ### Top 3 words in Elton John article

# In[55]:

john_word_count_sorted[0:3]


# ### Create a table and sort the tfidf's of Elton John article

# In[56]:

john_tfidf_table = john[['tfidf']].stack('tfidf', new_column_name = ['word', 'tfidf']).sort('tfidf', ascending = False)


# ### Top 3 tfidf words in Elton John article

# In[57]:

john_tfidf_table[0:3]


# ### Is Elton closer to Victoria Beckham?

# In[58]:

victoria = people[people['name'] == 'Victoria Beckham']


# In[60]:

graphlab.distances.cosine(john['tfidf'][0], victoria['tfidf'][0])


# ### Is Elton closer to Paul McCartney than Victoria?

# In[61]:

paul = people[people['name'] == 'Paul McCartney']


# In[62]:

graphlab.distances.cosine(john['tfidf'][0], paul['tfidf'][0])


# ### Build nearest neighbors model using word_count feature

# In[71]:

knn_word_count = graphlab.nearest_neighbors.create(people, features=['word_count'], label='name', distance='cosine')


# ### Build nearest neighbors model using tfidf feature

# In[72]:

knn_tfidf = graphlab.nearest_neighbors.create(people, features=['tfidf'], label='name', distance='cosine')


# ### Most similar article to Elton John using word_count knn model

# In[73]:

knn_word_count.query(john)


# ### Most similar article to Elton John using tfidf knn model

# In[74]:

knn_tfidf.query(john)


# ### Most similar article to Victoral Beckham using word_count knn model

# In[77]:

knn_word_count.query(victoria)


# ### Most similar article to Victoria Beckham using tfidf knn model

# In[78]:

knn_tfidf.query(victoria)

