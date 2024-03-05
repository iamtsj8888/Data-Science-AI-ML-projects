#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[62]:


dftest = pd.read_csv('Testing.csv')
dftrain = pd.read_csv('Training.csv')


# In[63]:


dftrain.head()


# In[64]:


dftest.head()


# In[65]:


dftrain.describe()


# In[66]:


dftrain.info()


# In[67]:


dftest.info()


# In[68]:


dftrain.shape


# In[69]:


dftest.shape


# In[70]:


dftrain.isnull().sum()


# In[71]:


dftest.isnull().sum()


# In[72]:


print(type(dftrain), type(dftest))


# In[73]:


dftrain.info()


# In[74]:


null_columns = dftrain.columns[dftrain.isnull().any()]
dftrain[null_columns].isnull().sum()


# In[75]:


dftrain.drop('Unnamed: 133', axis=1, inplace=True)
dftrain.info()


# In[76]:


dftest.info()


# In[77]:


null_columns=dftest.columns[dftest.isnull().any()]
dftest[null_columns].isnull().sum()


# In[78]:


columns = list(dftrain.columns)
sns.set(style="whitegrid")
colors = ['red', 'blue']
fig, axs = plt.subplots(nrows=len(columns), ncols=1,
figsize=(8, 3 * len(columns)))
for i, column in enumerate(columns):
    sns.countplot(x=column, data=dftrain, palette=colors, ax=axs[i])
    axs[i].set_title("Count of Symptom \"" + column + "\"", fontsize=14)
    axs[i].set_xlabel(column, fontsize=12)
    axs[i].set_ylabel("Count", fontsize=12)
    total = len(dftrain[column])
    for p in axs[i].patches:
        height = p.get_height()
        axs[i].text(p.get_x() + p.get_width() / 2., height + 0.1,
                    f'{height/total:.1%}', ha="center", fontsize=10)
plt.tight_layout()
plt.show()


# In[79]:


sorted(dftrain.prognosis.unique())


# In[80]:


dftest[dftest.duplicated(subset = None, keep = False)]


# In[81]:


dftrain.info()


# In[82]:


dftest.info()


# In[83]:


from collections import Counter
count = Counter(dftrain['prognosis'])
count.items()


# In[84]:


dftrain['prognosis'] = dftrain['prognosis'].astype('category')


plt.figure(figsize=(30, 5))
ax = sns.countplot(data=dftrain, x='prognosis', palette='PuBu')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.show()


# In[85]:


count = Counter(dftest['prognosis'])
count.items()


# In[86]:


dftest['prognosis'] = dftest['prognosis'].astype('category')
plt.figure(figsize=(30, 5))
ax = sns.countplot(data=dftest, x='prognosis', palette='PuBu')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.show()


# In[87]:


columns = list(dftrain.columns)
columns


# In[88]:


colors = ['red', 'green']

sns.set(style="whitegrid")

for i in columns:

    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=i, data=dftrain, palette=colors)

    ax.set_xlabel(f"{i} Absence/Presence", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Count of Symptom \"{i}\"", fontsize=16)

    
    plt.show()


# In[89]:


colors = ['#fffb08', '#fb08ff']
for i in columns:
    fig, ax = plt.subplots(figsize=(8, 6)) 
    bar = dftest.groupby(i).size().plot(kind='bar', color=colors, ax=ax)
    
    plt.xticks(rotation=0)
    plt.title("Count of Symptom \"" + i + "\"", fontsize=16)
    plt.xlabel("Presence of Symptom", fontsize=12)
    plt.ylabel("Count", fontsize=12)
  
    plt.legend(labels=['Absent', 'Present'], loc='upper right')

    for p in bar.patches:
        plt.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)
    
    plt.show()


# In[90]:


dftrain.describe()


# In[91]:


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import math

X_train = dftrain.iloc[:, :-1].values 
y_train = dftrain.iloc[:, 132].values
X_test = dftest.iloc[:, :-1].values
y_test = dftest.iloc[:, 132].values


# In[92]:


classifierDT = DecisionTreeClassifier(splitter='best', criterion='entropy', min_samples_leaf=2)
classifierDT.fit(X_train, y_train)


# In[93]:


classifierRF = RandomForestClassifier(criterion='entropy', min_samples_leaf=2)
classifierRF.fit(X_train, y_train)


# In[94]:


classifierMLP = MLPClassifier()
classifierMLP.fit(X_train, y_train)


# In[95]:


y_predMLP = classifierMLP.predict(X_test)
y_predDT = classifierDT.predict(X_test)
y_predRF = classifierRF.predict(X_test)


# In[96]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_predMLP))
print(classification_report(y_test, y_predMLP))

from sklearn.metrics import accuracy_score
print("Train Accuracy: ", accuracy_score(y_train, classifierMLP.predict(X_train)))
print("Test Accuracy: ", accuracy_score(y_test, y_predMLP))


# In[97]:


n_groups = 3
algorithms = ('Multilayer Perceptron (MLP) Neural Network', 'Decision Tree (DT)', 'Random Forest (RF)')
train_accuracy = (accuracy_score(y_train, classifierMLP.predict(X_train))*100, 
                  accuracy_score(y_train, classifierDT.predict(X_train))*100, 
                  accuracy_score(y_train, classifierRF.predict(X_train))*100)
test_accuracy = (accuracy_score(y_test, y_predMLP)*100, 
                 accuracy_score(y_test, y_predDT)*100, 
                 accuracy_score(y_test, y_predRF)*100)
fig, ax = plt.subplots(figsize=(15, 5))
index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.8
rects1 = plt.bar(index, train_accuracy, bar_width, alpha = opacity, color='Cornflowerblue', label='Train')
rects2 = plt.bar(index + bar_width, test_accuracy, bar_width, alpha = opacity, color='Teal', label='Test')
plt.xlabel('Algorithm') 
plt.ylabel('Accuracy (%)')
plt.ylim(0, 115)
plt.title('Comparison of Algorithm Accuracies')
plt.xticks(index + bar_width * 0.5, algorithms)
plt.legend(loc = 'upper right')
for index, data in enumerate(train_accuracy):
    plt.text(x = index - 0.035, y = data + 1, s = round(data, 2), fontdict = dict(fontsize = 8))
for index, data in enumerate(test_accuracy):
    plt.text(x = index + 0.25, y = data + 1, s = round(data, 2), fontdict = dict(fontsize = 8))
plt.show()


# In[98]:


imp = classifierDT.feature_importances_
imp


# In[99]:


columns = columns[:132]
column_names = ['symptom', 'importance']
df3 = np.vstack((columns, imp)).T
df3 = pd.DataFrame(df3, columns = column_names)
df3


# In[100]:


coefficients = classifierDT.feature_importances_

importance_threshold = np.quantile(coefficients, q = 0.75)
import plotly.express as px

fig = px.bar(x = coefficients, y = columns, orientation = 'h', color = coefficients, 
             color_continuous_scale = [(0, '#b7d2e8'), (1, '#295981')], labels = {'x': "Importance Value", 'y': "Feature"}, 
             title = "Feature Importance For Decision Tree Model")

fig.add_vline(x = importance_threshold, line_color = 'red', line_width = 0.8)
fig.add_vrect(x0 = importance_threshold, x1 = 0, line_width = 0, fillcolor = 'red', opacity = 0.2)
fig.show()


# In[101]:


import numpy
low_importance_features = numpy.array(df3.symptom[np.abs(coefficients) <= importance_threshold])
columns = list(low_importance_features)
columns


# In[102]:


for i in columns :
    dftrain.drop(i, axis=1, inplace=True)
    dftest.drop(i, axis=1, inplace=True)
dftrain.info()


# In[103]:


dftest.info


# In[104]:


X_train = dftrain.iloc[:, :-1].values
y_train = dftrain.iloc[:, 33].values
X_test = dftest.iloc[:, :-1].values
y_test = dftest.iloc[:, 33].values

classifierDT = DecisionTreeClassifier(splitter='best', criterion='entropy', min_samples_leaf=2)
classifierDT.fit(X_train, y_train)


# In[105]:


y_predDT = classifierDT.predict(X_test)

print(confusion_matrix(y_test, y_predDT))
print(classification_report(y_test, y_predDT))

print("Train Accuracy: ", accuracy_score(y_train, classifierDT.predict(X_train)))
print("Test Accuracy: ", accuracy_score(y_test, y_predDT))


# In[106]:


newdata = [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 
            0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1]]

probaDT = classifierDT.predict_proba(newdata)
probaDT.round(4)


# In[107]:


predDT = classifierDT.predict(newdata)
predDT


# In[ ]:




