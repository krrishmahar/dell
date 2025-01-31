#!/usr/bin/env python
# coding: utf-8

# In[99]:


import tensorflow as tf


# In[100]:


print(tf.__version__)


# ## Importing Liraries

# In[101]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[103]:


dataset=pd.read_csv('creditcard.csv')
# df = df.dropna().reset_index(drop=True)


# In[104]:


## Divide dataset to dependent and independent features
X=dataset.iloc[:, 1:-1]
y=dataset.iloc[:,-1]


# ### Split into train-test-split 

# In[105]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# ## Feature Scaling

# In[106]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[107]:


X_train[0]


# In[108]:


y_train


# In[109]:


X_test


# In[110]:


print(X_train.shape, X_test.shape)


# ## Part 2: Creating an ANN 

# In[111]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU,PReLU,ELU,ReLU
from tensorflow.keras.layers import Dropout


# In[112]:


# Initialize the ANN
classifier=Sequential()


# In[113]:


# Adding the Input layer
classifier.add(Dense(units=11, activation='relu'))


# In[114]:


# Adding the 1st hidden layer with dropout layer 
classifier.add(Dense(units=7))
classifier.add(Dropout(.3))


# In[115]:


classifier.add(Dense(units=6))
classifier.add(Dropout(0.5))


# In[116]:


classifier.add(Dense(units=1, activation='sigmoid'))


# In[117]:


classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[118]:


import tensorflow
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)


# In[119]:


# Early Stopping 
import tensorflow as tf
early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)


# In[120]:


model_history = classifier.fit(X_train,y_train,validation_split=0.33, batch_size=10, epochs=5, callbacks=early_stopping)


# In[121]:


model_history.history.keys()


# In[122]:


# summarize the history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')


# In[123]:


# summarize the history for accuracy
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')


# ## Predicting the Test values

# In[124]:


# Making the prediction and evaluating the model
y_pred=classifier.predict(X_test)
y_pred = (y_pred >= 0.5)


# In[125]:


# confusion matrix (WITHOUT DROPOUT LAYER)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm


# In[126]:


# confusion matrix(WITH DROPOUT LAYER)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm


# In[127]:


# calculate the accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
score


# ## Test input 

# In[143]:


new_input = np.array([-1.64, 1.52, -1.609850732,	3.997905588,	-0.522187865,	-1.426545319,	-2.537387306,	1.391657248,	-2.770089277,	-2.772272145,	3.202033207,	-2.899907388,	-0.595221881,	-4.289253782,	0.38972412,	-1.14074718,	-2.830055675,	-0.016822468,	0.416955705,	0.126910559,	0.517232371,	-0.035049369,	-0.465211076,	0.320198199,	0.044519167,	0.177839798,	0.261145003,	-0.143275875, 0])
output = np.array([1]) # value is True, bitch is Fraud


# In[144]:


X_test.shape, new_input.shape


# In[145]:


pred=classifier.predict(new_input.reshape(1,-1))
pred = (pred >= 0.5)


# In[146]:


pred


# In[147]:


# confusion matrix (WITHOUT DROPOUT LAYER)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(output, pred)
cm


# In[148]:


# confusion matrix(WITH DROPOUT LAYER)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(output, pred)
cm


# In[150]:


# calculate the accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(pred,output)
score


# In[151]:


classifier.save('ann_model.keras') 

