# Support Vector Machine (SVM)

###### DataTest02를 사용한 SVM

## 분류

### 1. 필요한 모듈을 Import한다.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split 
```

### CSV file을 불러와 데이터의 형태를 확인한다.

```
data = pd.read_csv("TestData02.csv",header=-1)
data
```

```python
	0		1		2		3
0	0.39	2.78	7.11	-8.07
1	1.65	6.70	2.42	12.24
2	5.67	6.38	3.79	23.96
3	2.31	6.27	4.80	4.29
4	3.67	6.67	2.38	16.37
```

### 다항의 변수로 이루어진 array와 그로 인한 분류 값의 array를 생성해 준다.

```python
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
```

### Series형태의 데이터를 ndarray의 형태로 변환해준다.

```python
x1 = x.values
y1 = y.values.reshape(-1,1)
```

### y value에 지정된 값을 0 또는 1의 분류값으로 변환해준다.

```python
for i in range(len(x)):
    if y1[i] >0:
        y1[i] =1
    else:
        y1[i] =0
```

### 변환된 데이터들을 확인한다.

```python
print(x1.shape), print(y.shape)
print(type(x1)), print(type(y1))
```

```python
(500, 3)
(500,)
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
(None, None)
```

### 훈련데이터와 검증제이터를 생성하여 러닝을 진행한다.

```python
xtrain, xtest, ytrain, ytest = train_test_split(x1,y1,test_size=0.25, random_state=0)
```

### 분류를 위한 머신으로 SVM을 선택 분류모델을 생성한다.

```python
svmClf= SVC(kernel="linear")
```

##### [kernel 참조](https://bskyvision.com/163)

### 훈련 데이터를 활용하여 학습시킨다.

```python
svmClf.fit(xtrain, ytrain)
```

```
C:\Users\Administrator\Anaconda3\lib\site-packages\sklearn\utils\validation.py:761: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
```

### 검증 데이터를 활용하여 검증을 실시한다.

```python
rResult = svmClf.predict(xtest)
```

```python
df = pd.DataFrame(ytest,rResult)
df.head()
```

```python
	 0
1.0	1.0
1.0	1.0
0.0	0.0
1.0	1.0
0.0	0.0
```



#### Machine성능 검증에 필요한 Module : metrics 을 설치

```python
from sklearn import metrics
```

```python
metrics.accuracy_score(ytest,rResult) #정확도 평가
```

```python
0.904
```

```python
metrics.recall_score(ytest,rResult) # Model Precision
```

```python
0.9791666666666666
```

```python
metrics.precision_score(ytest,rResult) # Model Recall
```

```python
0.9038461538461539
```

>분류 성능 평가
>
>- 분류 문제는 회귀 분석과 달리 다양한 성능 평가 기준이 필요하다.
>
>Scikit-Learn 에서 지원하는 분류 성능평가 명령 
>
>- sklearn.metrics 서브 패키지
>  - confusion_matrix()
>  - classfication_report()
>  - accuracy_score(y_true, y_pred)
>  - precision_score(y_true, y_pred)
>  - recall_score(y_true, y_pred)
>  - fbeta_score(y_true, y_pred, beta)
>  - f1_score(y_true, y_pred)
>        

##### [SVM 참조](https://datascienceschool.net/view-notebook/731e0d2ef52c41c686ba53dcaf346f32/)

