
# coding: utf-8

# # Predicting sentiment from product reviews
# 
# ### Fire up GraphLab Create

# In[58]:

import graphlab


# In[59]:

#Load and read some product review data of amazon baby products
products = graphlab.SFrame('amazon_baby.gl/')


# ### Explore the data

# In[60]:

products.head()


# ### Build the word count vector for each review

# In[61]:

products['word_count'] = graphlab.text_analytics.count_words(products['review'])


# In[62]:

products.head()


# In[63]:

graphlab.canvas.set_target('ipynb')


# In[64]:

products['name'].show()


# ### Examining the reviews for most-sold product 'Vulli Sophie the Giraffe Teether'

# In[65]:

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[66]:

len(giraffe_reviews)


# In[67]:

giraffe_reviews['rating'].show(view='Categorical')


# ### Build a sentiment classifier

# In[68]:

products['rating'].show(view='Categorical')


# ### Define what's a positive and a negative sentiment:
# 
# We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment. Reviews with a rating of 4 or higher will be considered positive, while the ones with rating of 2 or lower will have a negative sentiment. 

# In[69]:

#ignore all 3* reviews
products = products[products['rating'] != 3]


# In[70]:

#positive sentiment = 4* or 5* reviews
products['sentiment'] = products['rating'] >=4


# In[71]:

products.head()


# ### Train the sentiment classifier

# In[72]:

products[products['rating'] == 5]


# In[73]:

products[products['rating'] == 1]


# In[74]:

train_data,test_data = products.random_split(.8, seed=0)


# In[75]:

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)


# ### Evaluate the sentiment model

# In[76]:

sentiment_model.evaluate(test_data, metric='roc_curve')


# In[77]:

sentiment_model.show(view='Evaluation')


# ### Applying the learned model to understand sentiment for giraffe product

# In[78]:

giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')


# In[79]:

giraffe_reviews.head()


# ### Sort the reviews based on the predicted sentiment and explore

# In[80]:

giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)


# In[81]:

giraffe_reviews.head()


# ### Most positive reviews for the giraffe

# In[82]:

giraffe_reviews[0]['review']


# In[83]:

giraffe_reviews[1]['review']


# ### Most negative reviews for giraffe

# In[84]:

giraffe_reviews[-1]['review']


# In[85]:

giraffe_reviews[-2]['review']


# ## Build a model based on the below selected positive & negative words

# In[87]:

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']


# In[88]:

selected_words


# ### Write a python function to calculate no. of times words occur in word count of each review and explore

# In[142]:

#function to count frequency of selected words in each review
for word in selected_words:
    products[word] = products['word_count'].apply(lambda counts: counts.get(word, 0))    


# In[144]:

products.head()


# In[155]:

#count total occurences of selected words in products
d = {}
for word in selected_words:
    d[word] = products[word].sum()


# In[159]:

#max occurence of a selected word in products reviews
maximum = max(d, key=d.get)  
print(maximum, d[maximum])


# In[160]:

#min occurence of a selected word in products reviews
minimum = min(d, key=d.get)  
print(minimum, d[minimum])


# ### Split the product data with new columns

# In[145]:

train_data, test_data = products.random_split(.8, seed=0)


# ### Train the new sentiment selected words classifier

# In[172]:

selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=selected_words,
                                                     validation_set=test_data)


# In[147]:

selected_words_model['coefficients']


# In[162]:

selected_words_model['coefficients'].sort('value').print_rows(num_rows=12)


# ### Evaluate the selected words model

# In[164]:

selected_words_model.evaluate(test_data)


# In[166]:

#different classes/features of the products
d


# ### Train a majority class classfier

# In[170]:

majority_class = ['great']
majority_class_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=majority_class,
                                                     validation_set=test_data)


# ### Evaluate majority class model

# In[171]:

majority_class_model.evaluate(test_data)


# ### Examining the sentiments for baby trend diaper champ product

# In[183]:

diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']


# In[184]:

diaper_champ_reviews


# In[185]:

#Evaluate sentiments for reviews using the first sentiment model
diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews, output_type='probability')


# In[186]:

diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment', ascending=False)


# In[192]:

#most positive review for diaper champ after sorting
diaper_champ_reviews[0:1]


# In[193]:

#predict sentiment of most positive review using the second selected words model
selected_words_model.predict(diaper_champ_reviews[0:1], output_type='probability')


# ### This shows the sentiment model is most accurate than the selected words model and the majoriy class model
