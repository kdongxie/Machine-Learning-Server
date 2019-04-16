# WholeSale customers data 를 활용한 K-mean 함수

## Cluster의 개수를 구하라

## 준비

```python
import os
import graphviz
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pandas as pd
import numpy as np
import itertools 
import matplotlib.pyplot as plt 
from sklearn import svm, datasets 
# ! pip install scikit-plot
import scikitplot as skplt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.cluster import KMeans
```

### 데이터 호출

```python
data = pd.read_csv("Wholesale customers data.csv")
```

### 데이터 확인하기

```python
print(data.head())
# Channel(채널), Region(지역) : 볌주형
# Fresh (신선도), Milk (우유 양), Grocery(재료), ...: 연속형
print(data.info())
data.describe()
```

```python
   Channel  Region  Fresh  Milk  Grocery  Frozen  Detergents_Paper  Delicassen
0        2       3  12669  9656     7561     214              2674        1338
1        2       3   7057  9810     9568    1762              3293        1776
2        2       3   6353  8808     7684    2405              3516        7844
3        1       3  13265  1196     4221    6404               507        1788
4        2       3  22615  5410     7198    3915              1777        5185
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 440 entries, 0 to 439
Data columns (total 8 columns):
Channel             440 non-null int64
Region              440 non-null int64
Fresh               440 non-null int64
Milk                440 non-null int64
Grocery             440 non-null int64
Frozen              440 non-null int64
Detergents_Paper    440 non-null int64
Delicassen          440 non-null int64
dtypes: int64(8)
memory usage: 27.6 KB
None
```

![image](https://user-images.githubusercontent.com/46669551/55717724-e8ad9500-5a34-11e9-8a19-e59694613ba8.png)

### 데이터 결측치 확인

```python
data.isna().sum()
```

```python
Channel             0
Region              0
Fresh               0
Milk                0
Grocery             0
Frozen              0
Detergents_Paper    0
Delicassen          0
dtype: int64
```

## 데이터 처리

### 범주형 데이터와 연속형 데이터로 나누어 데이터를 분리

```python
x = data[['Channel','Region']]
y = data[['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']]
```

### 각각의 Value들만을 가져오기

```python
Y = x.values
X = y.values
```

### 데이터 확인

```python
X.shape, Y.shape
```

```python
((440, 6), (440, 2))
```

## KMean 함수

### K-Mean함수를 사용하여 Cluster 의 개수 구하기

```python
k = range(1,10)
data2 =[]
for i in k:
    #print(i)
    kmeanModel = KMeans(n_clusters=i).fit(X)
    kmeanModel.fit(X)
    t =cdist(X,kmeanModel.cluster_centers_,'euclidean') #
    #print(d3)
    t2 = sum(np.min(t, axis=1))/X.shape[0] #
    # k-평균 알고리즘(Elbow) 함수이다. 
    # 위의 코드는 k-평균 알고리즘의 함수를 구현한 코드이다.
    print(t2)
    data2.append(t2)
```

```python
14561.83033839006
12338.433138793233
10492.875588578885
9511.550221577088
8704.654299610598
7867.238703665478
7707.258343241413
7371.558777896475
7023.363036675074
```

```python
plt.plot(k, data2, 'bx-')
plt.show() 
```

![image](https://user-images.githubusercontent.com/46669551/55718278-527a6e80-5a36-11e9-9ce8-61f06b54d31a.png)

> 왼쫃 부터 각도의 변화가 가장 큰 지점의 값을 Cluster의 개수로 판단한다.