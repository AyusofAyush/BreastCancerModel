
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


from sklearn.datasets import load_breast_cancer


# In[5]:


cancer = load_breast_cancer()


# In[6]:


cancer


# In[7]:


df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns = np.append(cancer['feature_names'],['target']))


# In[8]:


df_cancer.head()


# In[11]:


X = df_cancer.drop(['target'], axis = 1)


# In[12]:


X.keys()


# In[13]:


X.head()


# In[14]:


y = df_cancer['target']


# In[15]:


y.shape


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# In[18]:


X_train.shape


# In[19]:


X_test.shape


# In[20]:


from sklearn.svm import SVC


# In[21]:


from sklearn.metrics import confusion_matrix, classification_report


# In[22]:


svc_model = SVC()


# In[23]:


svc_model.fit(X_train, y_train)


# In[24]:


y_pred = svc_model.predict(X_test)


# In[25]:


cm = confusion_matrix(y_test, y_pred)


# In[26]:


cm


# In[28]:


sns.heatmap(cm , annot = True)


# In[29]:


min_range = X_train.min()


# In[30]:


range_ = (X_train-min_range).max()


# In[31]:


X_train_scal = (X_train-min_range)/range_


# In[32]:


sns.scatterplot(x = X_train_scal['mean area'], y = X_train_scal['mean smoothness'], hue = y_train)


# In[34]:


min_test = X_test.min()
mn_range = (X_test-min_test).max()
X_test_scal = (X_test-min_test)/mn_range


# In[35]:


svc_model.fit(X_train_scal,y_train)


# In[36]:


y_pred2 = svc_model.predict(X_test_scal)


# In[37]:


cm =  confusion_matrix(y_test, y_pred2)


# In[38]:


cm


# In[39]:


sns.heatmap(cm, annot= True)


# In[40]:


print (classification_report(y_test, y_pred2))


# In[65]:


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 


# In[66]:


from sklearn.model_selection import GridSearchCV


# In[67]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[68]:


grid.fit(X_train_scal, y_train)


# In[69]:


grid.best_params_


# In[58]:


param_grid = {'C': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 'kernel': ['rbf']} 


# In[59]:


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4)


# In[60]:


grid.fit(X_train_scal, y_train)


# In[61]:


grid.best_params_


# In[70]:


grid_pred = grid.predict(X_test_scal)


# In[71]:


cm = confusion_matrix(y_test, grid_pred)


# In[72]:


sns.heatmap(cm, annot= True)


# In[73]:


print (classification_report(y_test, grid_pred))

