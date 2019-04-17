# 기차사고 확률

### 기본준비

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
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```

```python
train.head()
```

![image](https://user-images.githubusercontent.com/46669551/55714434-a3d23000-5a2d-11e9-8249-0dad466ebc27.png)

```python
test.head()
```

![image](https://user-images.githubusercontent.com/46669551/55714485-be0c0e00-5a2d-11e9-8d19-d14e2c1cec17.png)

### 데이터 요약

```python
train.describe()
```

![image](https://user-images.githubusercontent.com/46669551/55714546-d714bf00-5a2d-11e9-8724-fee447f73acf.png)

```python
test.describe()
```

![image](https://user-images.githubusercontent.com/46669551/55714623-ec89e900-5a2d-11e9-9f5a-2d048a58f4f0.png)

> 데이터를 확인해 보면 train데이터와 test데이터는 원래 하나의 데이터였다는 것을 알 수 있다.

### 결측치 (NA) 확인하기

```python
train.isna().sum()
```

```python
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
```

```python
test.isna().sum()
```

```python
PassengerId      0
Pclass           0
Name             0
Sex              0
Age             86
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64
```

### 결측치에 대해서 평균값을 넣어 준다.

```python
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)
```

### 재확인

```python
train.isna().sum(), test.isna().sum()
```

```python
(PassengerId      0
 Survived         0
 Pclass           0
 Name             0
 Sex              0
 Age              0
 SibSp            0
 Parch            0
 Ticket           0
 Fare             0
 Cabin          687
 Embarked         2
 dtype: int64, 
 
 PassengerId      0
 Pclass           0
 Name             0
 Sex              0
 Age              0
 SibSp            0
 Parch            0
 Ticket           0
 Fare             0
 Cabin          327
 Embarked         0
 dtype: int64)
```

## 변수들간의 관계를 알아보기

### 데이터의 Pclass에 대하여 Groupby하여, Class당 평균 생존 여부 확인하기

```python
# P-class, Survived
# Groupby 
data_temp = train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean()
```

```python
data_temp
```

![image](https://user-images.githubusercontent.com/46669551/55714971-a97c4580-5a2e-11e9-8650-b19d2433884a.png)

> 각각의 생존 가능성으로 해석 가능하다.

### 성별로 알아보는 생존률

```python
# Sex, Survived
data_temp2 = train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
```

```python
data_temp2
```

![image](https://user-images.githubusercontent.com/46669551/55715084-ee07e100-5a2e-11e9-90c5-5af76b4a8166.png)

> 열차 사고 시 남자보다 여성의 생존률이 더 높다.

### 동승자 수별 생존률

```python
data_temp3 = train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()
```

```python
data_temp3
```

![image](https://user-images.githubusercontent.com/46669551/55715162-214a7000-5a2f-11e9-88b9-45157c0a1d5a.png)

### 연령별 생존 분포를 히스토 그램으로 출력하기

```python
g = sns.FacetGrid(train,col='Survived')
g.map(plt.hist, 'Age', bins=20)
```

![image](https://user-images.githubusercontent.com/46669551/55715246-4f2fb480-5a2f-11e9-9b90-d48edda90832.png)

### Pclass와 Survived의 관계와 그 안에서의 연령 분포를 히스토 그램으로 나타내기

```python
grid = sns.FacetGrid(train, col='Survived', row="Pclass", size=2.5,aspect=2.5)
grid.map(plt.hist, 'Age',alpha=1, bins=20)
grid.add_legend()
```

![image](https://user-images.githubusercontent.com/46669551/55715538-ec8ae880-5a2f-11e9-94c9-ae67e5384e81.png)



## 다변량에 대한 관계를 알아볼 시

### 데이터 내부의 생존과 연관이 없는 데이터 들이 존재 할 시

```python
train.info()
# Name, Sex, Ticket, Cabin, Embarked 
# 생존과 관계가 없는 항목들이 존재 : Name, Ticket, Cabin(선실), Embarked (탑승한 곳)
train = train.drop(['Name', 'Cabin','Ticket','Embarked'], axis=1)
test = test.drop(['Name', 'Cabin','Ticket','Embarked'], axis=1)
```

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            891 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
```

### 성별로 이루어진 변수를 Labeling하여 숫자의 type으로 변환해 준다.

```python
labelSex = LabelEncoder() # 자동으로 Labeling을 해주는 함수
labelSex.fit(train['Sex'])
train['Sex'] = labelSex.transform(train['Sex'])
```

```python
train # male =1, Female =0
```

![image](https://user-images.githubusercontent.com/46669551/55715708-4a1f3500-5a30-11e9-8825-5b5d4b174de3.png)

### Survived (생존) 데이터에 대한 나머지 항목들의 관계성을 보기 위해 데이터를 나누어 준다.

```python
x = np.array(train.drop(['Survived'],1)).astype(float)
y = np.array(train['Survived'])
```

> y = a1x1+a2x2+...anxn 의 형식

```python
x.shape, y.shape
```

```python
((891, 7), (891,))
```

### x 데이터의 value값으로 DataFrame을 만들어 준다.

```python
pd.DataFrame(x).head()
```

![image](https://user-images.githubusercontent.com/46669551/55716334-b2224b00-5a31-11e9-862b-24eabf359fe8.png)

## K-Mean Cluster의 사용

```python
kmeans = KMeans(n_clusters=2, max_iter=600, algorithm='auto')
kmeans.fit(x)
```

```python
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
    n_clusters=2, n_init=10, n_jobs=None, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)
```

### x 데이터에 대하여 for문을 이용해 데이터 형식과 예측치를 뽑아준다.

```python
correct =0
for i in range(len(x)):
    predict_m = np.array(x[i].astype(float))
    print(predict_m.shape) # (7,) 의 형태
    predict_m = predict_m.reshape(-1,len(predict_m))
    print(predict_m.shape) # (1,7) 의 형태
    prediction = kmeans.predict(predict_m)
    print(prediction) # [0]또는 [1]로 이루어진 예측값 y의 형태
    if prediction[0] == y[i]: # 예측값과 실제 데이터가 같을 때 correct의 변수에 1을 더해준다. 
        correct +=1 # 사망 발생 수
```

### 사망확률을 계산해준다.

```python
print(correct/len(x)) # 전체 사망 수 / 사고 횟 수 = 사망 확률
```

```
0.49158249158249157
```

## 데이터의 범위가 너무 클 경우 Scaleing 

### x 데이터에 정의 된 값의 범위가 너무 큰 경우 정확도를 높히기 위해 Scale 조절을 수행할 수 있다.

```python
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
```

```python
x_scaled 
# 범위 0~ 1 로 바꾸어 준다. 
# 범위가 큰 데이터를 균일한 데이터 값으로 변환함에 목적을 둔다.
```

```python
array([[0.        , 1.        , 1.        , ..., 0.125     , 0.        ,
        0.01415106],
       [0.0011236 , 0.        , 0.        , ..., 0.125     , 0.        ,
        0.13913574],
       [0.00224719, 1.        , 0.        , ..., 0.        , 0.        ,
        0.01546857],
       ...,
       [0.99775281, 1.        , 0.        , ..., 0.125     , 0.33333333,
        0.04577135],
       [0.9988764 , 0.        , 1.        , ..., 0.        , 0.        ,
        0.0585561 ],
       [1.        , 1.        , 1.        , ..., 0.        , 0.        ,
        0.01512699]])
# 전체의 데이터가 0~1 사이의 값으로 변환됨을 알 수 있다.
```

### 재 측정 수행 

```python
correct2 =0
for i in range(len(x_scaled)):
    predict_m2 = np.array(x_scaled[i].astype(float))
    print(predict_m)
    predict_m2 = predict_m2.reshape(-1,len(predict_m2))
    print(predict_m2)
    prediction2 = kmeans.predict(predict_m2)
    print(prediction2)
    if prediction2[0] == y[i]:
        correct2 +=1
```

### Scaled 된 데이터에 의한 사망 확률

```python
print(correct2/len(x_scaled)) 
# Scaled 된 데이터의 사망 발생 확률
```

```python
0.6161616161616161
```

