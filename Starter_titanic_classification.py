
# coding: utf-8

# In[204]:


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


# In[205]:


# Read train and test data with pd.read_csv():
train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")


# In[206]:


# copy data in order to avoid any change in the original:
train = train_data.copy()
test = test_data.copy()


# In[207]:


train.head()


# In[208]:


sns.barplot(x = 'Pclass', y = 'Survived', data = train);


# In[209]:


plt.show()


# In[210]:


sns.barplot(x = 'Sex', y = 'Survived', data = train);


# In[155]:


plt.show()


# ## Outlier Treatment

# In[156]:


train.describe().T


# In[157]:


sns.boxplot(x = train['Fare']);


# In[158]:


plt.show()


# In[159]:


Q1 = train['Fare'].quantile(0.25)
Q3 = train['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1- 1.5*IQR
lower_limit

upper_limit = Q3 + 1.5*IQR
upper_limit


# In[160]:


train.sort_values("Fare", ascending=False).head()


# In[161]:


# In boxplot, there are too many data higher than upper limit; we can not change all. Just repress the highest value -512- 
train['Fare'] = train['Fare'].replace(512.3292, 300)


# In[162]:


train.sort_values("Fare", ascending=False).head()


# In[163]:


# We can drop the Ticket feature since it is unlikely to have useful information
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)


# ## Missing value treatment

# ### Missing treatment in Age

# In[164]:


train.isnull().sum()


# In[165]:


train["Age"] = train["Age"].fillna(train["Age"].mean())


# In[166]:


test["Age"] = test["Age"].fillna(test["Age"].mean())


# In[167]:


train.isnull().sum()


# ### Missing treatment in embarked

# In[168]:


train["Embarked"].value_counts()


# In[169]:


# Fill NA with the most frequent value:
train["Embarked"] = train["Embarked"].fillna("S")


# In[170]:


test["Embarked"] = test["Embarked"].fillna("S")


# ### Missing treatment in Fare

# In[171]:


train.isnull().sum()


# In[172]:


test.isnull().sum() 


# In[173]:


test[test["Fare"].isnull()]


# In[174]:


test[["Pclass","Fare"]].groupby("Pclass").mean()


# In[175]:


test["Fare"] = test["Fare"].fillna(12)


# ### Missing treatment in Cabin

# In[176]:


# Create CabinBool variable which states if someone has a Cabin data or not:

train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)

train.head()


# In[177]:


train.isnull().sum()


# ### Variable transformation

# In[178]:


embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)


# In[179]:


train.head()


# ### Sex - 1/0

# In[180]:


# Convert Sex values into 1-0:

from sklearn import preprocessing

lbe = preprocessing.LabelEncoder()
train["Sex"] = lbe.fit_transform(train["Sex"])
test["Sex"] = lbe.fit_transform(test["Sex"])


# In[181]:


train.head()


# ### Name - Title 

# In[182]:


train["Title"] = train["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
test["Title"] = test["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[183]:


train.head()


# In[184]:


train['Title'] = train['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
train['Title'] = train['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')


# In[185]:


test['Title'] = test['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
test['Title'] = test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')


# In[186]:


train.head()


# In[187]:


train[["Title","PassengerId"]].groupby("Title").count()


# In[188]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[189]:


# Map each of the title groups to a numerical value

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5}

train['Title'] = train['Title'].map(title_mapping)


# In[190]:


train.isnull().sum()


# In[191]:


test['Title'] = test['Title'].map(title_mapping)


# In[192]:


test.head()


# In[193]:


train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# ### Age group transformation 

# In[194]:


train.head()


# In[195]:


bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = mylabels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = mylabels)


# In[196]:


# Map each Age value to a numerical value:
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)


# In[197]:


train.head()


# In[198]:


#dropping the Age feature for now, might change:
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)


# ### Fare group

# In[199]:


# Map Fare values into groups of numerical values:
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])


# In[200]:


# Drop Fare values:
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


# ## Feature Engineering

# In[201]:


train.head() 


# In[202]:


train["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1


# In[134]:


test["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1


# In[135]:


# Create new feature of family size:

train['Single'] = train['FamilySize'].map(lambda s: 1 if s == 1 else 0)
train['SmallFam'] = train['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
train['MedFam'] = train['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
train['LargeFam'] = train['FamilySize'].map(lambda s: 1 if s >= 5 else 0)


# In[136]:


train.head()


# In[137]:


# Create new feature of family size:

test['Single'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)
test['SmallFam'] = test['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
test['MedFam'] = test['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
test['LargeFam'] = test['FamilySize'].map(lambda s: 1 if s >= 5 else 0)


# In[138]:


test.head() 


# ### Emarked - Title

# In[139]:


# Convert Title and Embarked into dummy variables:

train = pd.get_dummies(train, columns = ["Title"])
train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")


# In[140]:


train.head()


# In[141]:


test = pd.get_dummies(test, columns = ["Title"])
test = pd.get_dummies(test, columns = ["Embarked"], prefix="Em")


# In[142]:


test.head() 


# ### PClass 

# In[143]:


# Create categorical values for Pclass:
train["Pclass"] = train["Pclass"].astype("category")
train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc")


# In[144]:


test["Pclass"] = test["Pclass"].astype("category")
test = pd.get_dummies(test, columns = ["Pclass"],prefix="Pc")


# In[145]:


train.head()


# In[146]:


test.head()


# In[147]:


train.to_csv('./data/train_transformed.csv')


# In[148]:


test.to_csv('./data/test_transformed.csv')

