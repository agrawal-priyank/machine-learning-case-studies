
# coding: utf-8

# ### Image Retrieval with Deep Features
# 
# 
# ### Fire up GraphLab Create

# In[1]:

import graphlab


# ### Load the CIFAR-10 dataset
# 
# Its a popular benchmark dataset in computer vision called CIFAR-10 reduced to just 4 categories = {'cat','bird','automobile','dog'}.

# In[2]:

image_train = graphlab.SFrame('image_train_data/')


# ### Computing deep features for our images
# 
# The two lines below to compute deep features. This computation takes a little while, so its already presenet as a column in the data loaded.

# In[3]:

#deep_learning_model = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')
#image_train['deep_features'] = deep_learning_model.extract_features(image_train)


# In[8]:

image_train.head()


# ### Train a nearest-neighbors model for retrieving images using deep features

# In[9]:

knn_model = graphlab.nearest_neighbors.create(image_train,features=['deep_features'],
                                             label='id')


# ### Use image retrieval model with deep features to find similar images
# 
# Let's find similar images to this cat picture.

# In[10]:

graphlab.canvas.set_target('ipynb')
cat = image_train[18:19]
cat['image'].show()


# In[11]:

knn_model.query(cat)


# In[12]:

def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'],'id')


# In[13]:

cat_neighbors = get_images_from_ids(knn_model.query(cat))


# In[14]:

cat_neighbors['image'].show()


# ### Finding similar images to a car

# In[15]:

car = image_train[8:9]
car['image'].show()


# In[16]:

get_images_from_ids(knn_model.query(car))['image'].show()


# ### Lambda to find and show nearest neighbor images

# In[17]:

show_neighbors = lambda i: get_images_from_ids(knn_model.query(image_train[i:i+1]))['image'].show()


# In[18]:

show_neighbors(8)


# In[19]:

show_neighbors(26)


# ### Summary of the train data

# In[28]:

image_train['label'].sketch_summary()


# ### Split the train data according to different labels

# In[38]:

image_train_auto = image_train[image_train['label'] == 'automobile']


# In[40]:

image_train_cat = image_train[image_train['label'] == 'cat']


# In[41]:

image_train_dog = image_train[image_train['label'] == 'dog']


# In[42]:

image_train_bird = image_train[image_train['label'] == 'bird']


# ### Train a nearest-neighbors model for each of the labels

# In[44]:

auto_model = graphlab.nearest_neighbors.create(image_train_auto, features = ['deep_features'], label = 'id')


# In[48]:

cat_model = graphlab.nearest_neighbors.create(image_train_cat, features = ['deep_features'], label = 'id')


# In[49]:

dog_model = graphlab.nearest_neighbors.create(image_train_dog, features = ['deep_features'], label = 'id')


# In[50]:

bird_model = graphlab.nearest_neighbors.create(image_train_bird, features = ['deep_features'], label = 'id')


# ### So basicaly now we have four models to retrieve different category of images from any dataset of choice

# ### Loading the test data

# In[51]:

image_test = graphlab.SFrame('image_test_data/')


# In[54]:

image_test[0:1].show()


# ### Nearest cat from the image_train data to the cat above

# In[63]:

nearest_cats = cat_model.query(image_test[0:1])


# In[66]:

nearest_cats


# ### The nearest cat has the reference label as 16289

# In[68]:

get_images_from_ids(nearest_cats[nearest_cats['reference_label'] == 16289])['image'].show()


# ### Nearest dog from the image_train data to the cat above

# In[74]:

nearest_dogs = dog_model.query(image_test[0:1])


# In[75]:

nearest_dogs


# ### The nearest dog has the reference label as 16979

# In[76]:

get_images_from_ids(nearest_dogs[0:1])['image'].show()


# ### Mean distance between the cat and its first 5 neighboring cats

# In[86]:

nearest_cats['distance'].mean()


# ### Mean distance between the cat and its first 5 neighboring dogs

# In[87]:

nearest_dogs['distance'].mean()


# ### Split the test data according to different labels

# In[88]:

image_test_auto = image_test[image_test['label'] == 'automobile']


# In[89]:

image_test_cat = image_test[image_test['label'] == 'cat']


# In[90]:

image_test_dog = image_test[image_test['label'] == 'dog']


# In[91]:

image_test_bird = image_test[image_test['label'] == 'bird']


# ### Single closest neighbour to dog from all the trained models

# In[95]:

dog_auto_neighbors = auto_model.query(image_test_dog, k=1)


# In[94]:

dog_cat_neighbors = cat_model.query(image_test_dog, k=1)


# In[96]:

dog_dog_neighbors = dog_model.query(image_test_dog, k=1)


# In[97]:

dog_bird_neighbors = bird_model.query(image_test_dog, k=1)


# ### Storing distances of the dog from each of its neighbours found in a different frame

# In[108]:

dog_distances = graphlab.SFrame({'dog-auto': dog_auto_neighbors['distance'], 'dog-cat' : dog_cat_neighbors['distance'], 
                                 'dog-dog' : dog_dog_neighbors['distance'], 'dog-bird' : dog_bird_neighbors['distance']})


# In[109]:

dog_distances


# In[117]:

dog_distances['dog-cat'][2]


# In[123]:

def is_dog_correct(row):
    d = row['dog-dog']
    if d > row['dog-auto'] or d > row['dog-cat'] or d > row['dog-bird']:
        return 0
    else:
        return 1    


# ### Total number of correctly classified dogs from the image test dataset

# In[130]:

total_correct = dog_distances.apply(is_dog_correct).sum()


# In[131]:

total_correct


# ### Accuracy correcly classified dogs from the dog image test dataset

# In[132]:

total_number = len(image_test_dog)


# In[134]:

( float(total_correct) * 100 ) / float(total_number)


# ### Total Accuracy is 67.8%
