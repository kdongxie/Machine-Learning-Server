# Multiple Linear Regression 

## y = b0 + b1x1 + b2x2 + ... + e (가중치)

### WineQuality.csv File을 이용한 다중선형회귀분석

#### 1. 필요한 Packages를 Import시켜 준다.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as seabornInstance 
#seaborn은 Matplotlib을 기반으로 다양한 색상 테마와 통계용 차트 등의 기능을 추가한 시각화 패키지이다. 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# 교차검증을 하려면 두 종류의 데이터 집합이 필요하다.
# - 모형 추정 즉 학습을 위한 데이터 집합 (training data set)
# - 성능 검증을 위한 데이터 집합 (test data set)
# 두 데이터 집합 모두 종속 변수값이 있어야 한다. 따라서 보통은 가지고 있는 데이터 집합을 학습용과 검증용으로 나누어 학습용 데이터만을 사용하여 회귀분석 모형을 만들고 검증용 데이터로 성능을 계산하는 학습/검증 데이터 분리(train-test split) 방법을 사용한다.
```

#### 2. WineQuality.csv File 을 호출한다.

```python
data = pd.read_csv("WineQuality.csv")
data.head()
```

![image](https://user-images.githubusercontent.com/46669551/55400375-b4475e00-5588-11e9-9d5b-6ae9182ed388.png)

#### 3. 데이터의 형식과 Describe를 통해 요약본을 출력한다.

```python
data.shape
data.describe()
```

```
(1599, 12)
```

![image](https://user-images.githubusercontent.com/46669551/55400699-8b739880-5589-11e9-9d9f-7816de80009e.png)

#### 4. Data 내부에 존재할지 모르는 Null값을 찾고 만약 존재한다면 값을 넣어주는 작업을 진행

```python
print(data.isnull().any()) # Null값 존재 여부
data = data.fillna(method = 'ffill') # Null값에 값을 채워 넣어라
```

```
fixed acidity           False
volatile acidity        False
citric acid             False
residual sugar          False
chlorides               False
free sulfur dioxide     False
total sulfur dioxide    False
density                 False
pH                      False
sulphates               False
alcohol                 False
quality                 False
dtype: bool
```

#### 5. Data를 나누어 변수에 지정해 준다

```python
x= data.iloc[:,:-1].values # 전체 row에 끝에서 두번째 까지의 colums를 가져온다
print(x)
y= data.iloc[:,-1:].values # 전체 row에 끝에 위치한 column을 가져온다
print(y)
```

```
[[ 7.4    0.7    0.    ...  3.51   0.56   9.4  ]
 [ 7.8    0.88   0.    ...  3.2    0.68   9.8  ]
 [ 7.8    0.76   0.04  ...  3.26   0.65   9.8  ]
 ...
 [ 6.3    0.51   0.13  ...  3.42   0.75  11.   ]
 [ 5.9    0.645  0.12  ...  3.57   0.71  10.2  ]
 [ 6.     0.31   0.47  ...  3.39   0.66  11.   ]]

[[5]
 [5]
 [5]
 ...
 [6]
 [5]
 [6]]
```

#### 6. 두개의 변수로 저장된 Data의 형태를 출력해본다.

```python
x.shape, y.shape
```

```
((1599, 11), (1599, 1))
```

#### 7. seaborn Package를 이용하여 data['quality']의 누적빈도를 그래프로 출력한다.

```python
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(data['quality'])
plt.show()
```

![image](https://user-images.githubusercontent.com/46669551/55400989-5c115b80-558a-11e9-8d4b-f21245876033.png)

#### 8. 교차검증을 수행한다.

##### 1. 학습데이터와 검증데이터를 생성한다.

```python
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=0)
```

##### 2. 선형회귀함수를 통해 학습데이터를 fit한다.

````
regressionTest = LinearRegression()
regressionTest.fit(xtrain,ytrain)
````

##### 3. 검증데이터 xtest의 예측값을 구한다

```python
result = regressionTest.predict(xtest)
result
```

```
array([[5.7829301 ],
       [5.03619267],
       [6.59698929],
       [5.33912637],
       [5.93952898],
       [5.0072068 ],
       [5.39616171],
       [6.05211188],
       [4.86760343],
       [4.95067572],
       [5.28580441],
       [5.41265269],
       [5.7057424 ],
       [5.12921737],
       [5.52885206],
       [6.38052412],
       [6.81012527],
       [5.73803346],
       [5.97618825],
       [5.08613415],
       [6.34479863],
       [5.16400983],...
```

##### 4. 가중치 값을 출력한다.  .coef_

```python
regressionTest.coef_ # e 값 
```

```
array([[ 4.12835075e-02, -1.14952802e+00, -1.77927063e-01,
         2.78700036e-02, -1.87340739e+00,  2.68362616e-03,
        -2.77748370e-03, -3.15166657e+01, -2.54486051e-01,
         9.24040106e-01,  2.67797417e-01]])
```

##### 5. 검증데이터인 ytest와 학습데이터의 예측치를 DataFrame의 형태로 나타내어 준다.

```python
#ytest, result
df2 = pd.DataFrame({'ytest':ytest.flatten(),"pre":result.flatten()})
df2.head()
```

![image](https://user-images.githubusercontent.com/46669551/55401362-71d35080-558b-11e9-9dd6-fcc091069921.png)

##### 6. 예측 모델의 성능 측정하기

```python
np.sqrt(mean_squared_error(ytest, result))
```

```
0.6200574149384263
```

>
>
>### 상관계수(Correlation coefficient)는 두 변수간의 연관된 정도를 나타냄.
>
>
>
>다음 둘다 오차율을 나타내는 값임.
>
>## (둘다 오차의 정도에대한 값이므로, 당연히 0에 가까울수록 좋음)
>
>
>
>**Root Mean Square Error (RMSE) : 편차 제곱의 평균에 루트를 씌운 값.**
>
>**이걸 기준으로 성능을 올리면, 이는 표준편차를 기준으로 하기때문에, 큰 에러를 최대한 줄이는 방향으로 학습을 함.**
>
>-> ex) 정답이 9인 경우
>
>9, 9, 6, 9 보다 
>
>8, 8, 8 ,8 를 좋게 평가
>
>
>
>**mean absolute error (MAE) : 편차에 절대값을 씌운것의 평균**
>
>**단순 편차의 절대값의 평균임. 그러므로 RMSE와 달리 작은 에러에 더 민감함.**
>
>**-> ex) 정답이 9인 경우8, 8, 8 ,8 보다 9, 9, 6, 9 를 좋게 평가**