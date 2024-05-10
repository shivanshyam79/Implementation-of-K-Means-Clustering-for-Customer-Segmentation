# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## step 1. Import the required packages.
## step 2. Import the dataset to work on.
## step 3. From sklearn module import kmeans.
## step 4. Define number of clusters to be made.
## step 5. Assign the cluster values.
## step 6. Plot the cluster using matplotlib.pyplot
## step 7. End the program
## step 7 stop.
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: shyam R
RegisterNumber:  212223040200
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i,init="k-means++")
  kmeans.fit(data.iloc[:,3:])
  wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("no of clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
km = KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
y_pred = km.predict(data.iloc[:,3:])

data["cluster"] =y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4= data[data["cluster"]==4]

plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="blue",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="pink",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="green",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="red",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="orange",label="cluster4")
plt.title("Customer Segment")
plt.legend()

*/
```

## Output:
![firsst](https://github.com/shivanshyam79/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/151513860/d7c11644-46ce-48ee-a62a-4fa50f68908e)
![middlee](https://github.com/shivanshyam79/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/151513860/ee408b97-1562-4bf8-a526-f9d208470d3d)
![lasst](https://github.com/shivanshyam79/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/151513860/09147232-d6e2-4b39-97d5-ae532b89cf7c)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
