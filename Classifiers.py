
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd

# data visualization libraries:
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')

# to display all columns:
pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[6]:


train = pd.read_csv('./data/train_transformed.csv')
test = pd.read_csv('./data/test_transformed.csv')


# In[7]:


predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20, random_state = 0)


# In[8]:


x_train.shape


# In[9]:


x_test.shape


# ## Logistic Regression

# In[10]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_logreg)


# ## Random Forest

# In[11]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_test)
acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_randomforest)


# ## Gradient Boosting Classifier

# In[12]:


from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_test)
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gbk)


# In[13]:


xgb_params = {
        'n_estimators': [200, 500],
        'subsample': [0.6, 1.0],
        'max_depth': [2,5,8],
        'learning_rate': [0.1,0.01,0.02],
        "min_samples_split": [2,5,10]}


# In[14]:


xgb = GradientBoostingClassifier()

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)


# In[15]:


xgb_cv_model.fit(x_train, y_train)


# In[16]:


xgb = GradientBoostingClassifier(learning_rate = xgb_cv_model.best_params_["learning_rate"], 
                    max_depth = xgb_cv_model.best_params_["max_depth"],
                    min_samples_split = xgb_cv_model.best_params_["min_samples_split"],
                    n_estimators = xgb_cv_model.best_params_["n_estimators"],
                    subsample = xgb_cv_model.best_params_["subsample"])


# In[17]:


xgb_tuned =  xgb.fit(x_train,y_train)


# In[18]:


y_pred = xgb_tuned.predict(x_test)
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gbk)


# # Prediction 

# In[20]:


#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = xgb_tuned.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('predictions.csv', index=False)

