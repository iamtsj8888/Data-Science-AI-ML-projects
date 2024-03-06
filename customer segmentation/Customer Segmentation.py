#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('Mall_Customers.csv')


# In[4]:


df.head(5)


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df = df.drop('CustomerID', axis=1)


# In[10]:


X = df[['Annual Income (k$)', 'Spending Score (1-100)']]


# In[11]:


X


# In[14]:


plt.figure(figsize=(10, 6))
sns.scatterplot(X, x= "Annual Income (k$)", y= "Spending Score (1-100)")
plt.show()


# In[19]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[21]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
print(wcss)


# In[22]:


plt.figure(figsize=(10, 6)) 
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')  
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(range(1, 11))  
plt.grid(True) 
plt.show()


# In[23]:


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(X_scaled)


# In[24]:


plt.figure(figsize=(10, 8))
for cluster_label in range(5):  
    cluster_points = X[kmeans.labels_ == cluster_label]
    centroid = cluster_points.mean(axis=0)  
    plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'],
                s=50, label=f'Cluster {cluster_label + 1}')  
    plt.scatter(centroid[0], centroid[1], s=300, c='black', marker='*', label=f'Centroid {cluster_label + 1}')  
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# The result of the analysis shows that the retail store customers can be grouped into 5 clusters or segments for targeted marketing.
# 
# Cluster 1 (Blue): These are low-income earning customers with high spending scores. I can assume that why this group of customers spend more at the retail store despite earning less is because they enjoy and are satisfied with the services rendered at the retail store.
# 
# Cluster 2 (Orange): This group of customers have a higher income but they do not spend more at the store. One of the assumptions could be that they are not satisfied with the services rendered at the store. They are another ideal group to be targeted by the marketing team because they have the potential to bring in increased profit for the store.
# 
# Cluster 3 (Green): The customers in this group are high-income earners with high spending scores. They bring in profit. Discounts and other offers targeted at this group will increase their spending score and maximize profit.
# 
# Cluster 4 (Red): These are average income earners with average spending scores. They are cautious with their spending at the store.
# 
# Cluster 5 (Purple): Low-income earners with a low spending score. I can assume that this is so because people with low income will tend to purchase fewer items at the store.

# In[ ]:




