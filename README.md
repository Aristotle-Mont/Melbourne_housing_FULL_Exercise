# Melbourne_housing_FULL_Exercise
---

## Melbourne Housing Data Analysis

This repository provides a comprehensive analysis of the Melbourne Housing Market dataset. The aim is to process the data, perform exploratory data analysis, and attempt clustering to understand inherent groupings within the data.

---

### Data Overview

The dataset contains various information about houses in Melbourne including their price, location, number of rooms, type of house, etc.

Initial Analysis:
```python
import pandas as pd

data = pd.read_csv('/content/Melbourne_housing_FULL.csv')
print(data.head())
```

---

### Data Preprocessing

**Handling Missing Values**:
```python
data = data.dropna(subset=['Price'])
```

**Descriptive Analysis**:
```python
data.describe(include='all') 
```

**Data Transformations**:
1. **Log Transformation** for 'Price' column:
```python
import numpy as np

data['LogPrice'] = np.log(data['Price'])
```

2. **Date Transformation**:
```python
data['Year'] = pd.to_datetime(data['Date']).dt.year
data['Month'] = pd.to_datetime(data['Date']).dt.month
```

3. **Categorical Variables Transformation**:
```python
data = pd.get_dummies(data, columns=['Suburb', 'CouncilArea'], drop_first=True)
print(data.columns)
```

---

### Data Visualization

Using `matplotlib` and `seaborn` for visualization.

**Histogram for Price Distribution**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data['Price'])
plt.show()
```

**Scatter and Box Plots for Price by different Postcodes**:
```python
# Code as mentioned above
```

---

### Clustering Strategy

1. **Understanding the Data**
2. **Data Preprocessing**
3. **Feature Selection**
4. **Choosing a Clustering Algorithm**: KMeans, DBSCAN, Agglomerative Clustering.
5. **Training the Model**
6. **Evaluate & Visualize Clusters**
7. **Interpretation**
8. **Documentation & Presentation**

**Loading & Preprocessing**:
```python
# Code as mentioned above
```

**Finding optimal number of clusters**:
```python
# Code for Elbow Method
```

**Train the model and Assign Clusters**:
```python
k = 3 
kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
data['Cluster'] = kmeans.labels_
```

**2D Visualization using PCA**:
```python
# Code for PCA visualization
```

---

### Interpretation

Post clustering, we can inspect each cluster to identify its defining characteristics:

```python
print(data.groupby('Cluster').mean())
```

For example, from the provided sample output:
- **Cluster 0**: Houses are typically larger (3 rooms) and are slightly older (around 61 years).
- **Cluster 1**: Smaller houses (2 rooms) that are newer (around 47 years).
- **Cluster 2**: Large houses (3.76 rooms) with the highest age (around 52 years).

---

### Conclusion

This analysis provides a systematic approach to understanding the Melbourne housing market. Clustering helps in identifying inherent groupings which can be leveraged for various business use-cases including targeted marketing, prediction modeling, etc.

