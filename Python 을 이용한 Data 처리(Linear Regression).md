# Python 을 이용한 Data 처리

## Pandas, Numpy, Matplotlib, sklearn

### 활용

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Machine Learning Algorism
# Machine Learning Packages : sklearn
from sklearn.model_selection  import train_test_split 
from sklearn.linear_model import LinearRegression   
```

### pandas를 이용하여 csv File을 호출하여 출력한다.

```python
data2 = pd.read_csv("salary.csv")
data2.head(10)
```

```
YearsExperience	Salary
0	1.1	39343.0
1	1.3	46205.0
2	1.5	37731.0
3	2.0	43525.0
4	2.2	39891.0
5	2.9	56642.0
6	3.0	60150.0
7	3.2	54445.0
8	3.2	64445.0
9	3.7	57189.0
```

### Data의 value만을 가져온다

```python
x = data2["YearsExperience"].values
y = data2["Salary"].values
print(x), print(y)
print(x.shape), print(y.shape)
print(type(x)), print(type(y))
```

```
# "YearsExperience"와 "Salary" value들만 가져옴
[ 1.1  1.3  1.5  2.   2.2  2.9  3.   3.2  3.2  3.7  3.9  4.   4.   4.1
  4.5  4.9  5.1  5.3  5.9  6.   6.8  7.1  7.9  8.2  8.7  9.   9.5  9.6
 10.3 10.5]
 
 [ 39343.  46205.  37731.  43525.  39891.  56642.  60150.  54445.  64445.
  57189.  63218.  55794.  56957.  57081.  61111.  67938.  66029.  83088.
  81363.  93940.  91738.  98273. 101302. 113812. 109431. 105582. 116969.
 112635. 122391. 121872.]
 
# 1차원의 형태로 전체 값은 30개
 (30,) 
 (30,)
 
# ndarray의 형태
<class 'numpy.ndarray'> 
<class 'numpy.ndarray'>
```

### 각각의 원소로 나누어 준다 (형태변환)

```python
x1 = x.reshape(len(x),1)
y1 = y.reshape(-1,1)
print(x1), print(y1)
print(type(x1)),print(type(y1))
```

```
# List의 형태는 같지만 각각의 value들의 값으로 이루어진 list로 변환
[[ 1.1]
 [ 1.3]
 [ 1.5]
 [ 2. ]
 [ 2.2]
 [ 2.9]
 [ 3. ]
 [ 3.2]
 [ 3.2]
 [ 3.7]
 [ 3.9]
 [ 4. ]
 [ 4. ]
 [ 4.1]
 [ 4.5]
 [ 4.9]
 [ 5.1]
 [ 5.3]
 [ 5.9]
 [ 6. ]
 [ 6.8]
 [ 7.1]
 [ 7.9]
 [ 8.2]
 [ 8.7]
 [ 9. ]
 [ 9.5]
 [ 9.6]
 [10.3]
 [10.5]]
 
[[ 39343.]
 [ 46205.]
 [ 37731.]
 [ 43525.]
 [ 39891.]
 [ 56642.]
 [ 60150.]
 [ 54445.]
 [ 64445.]
 [ 57189.]
 [ 63218.]
 [ 55794.]
 [ 56957.]
 [ 57081.]
 [ 61111.]
 [ 67938.]
 [ 66029.]
 [ 83088.]
 [ 81363.]
 [ 93940.]
 [ 91738.]
 [ 98273.]
 [101302.]
 [113812.]
 [109431.]
 [105582.]
 [116969.]
 [112635.]
 [122391.]
 [121872.]]
 
(30, 1)
(30, 1)

<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
```

### Training 변수와 Test변수를 선언해준다

```python
xTrain, xTest, yTrain, yTest = train_test_split(x1,y1,test_size=1/3,random_state=0)
# 전체 데이터의 1/3만을 사용하여 test를 수행하게 된다.
```

### 선형회기함수인 LinearRegression을 변수에 지정해 준다

```python
linearTest = LinearRegression()
```

### Traning 변수의 x값과 y값을 합쳐준다

```python
linearTest.fit(xTrain,yTrain)
```

```
# 출력
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
         normalize=False)
```

### .predict를 사용하여 x값에 대한 y의예측 값을 만들기

```python
yprediction = linearTest.predict(xTest)
print(yprediction)
```

```
[[ 40835.10590871]
 [123079.39940819]
 [ 65134.55626083]
 [ 63265.36777221]
 [115602.64545369]
 [108125.8914992 ]
 [116537.23969801]
 [ 64199.96201652]
 [ 76349.68719258]
 [100649.1375447 ]]
 # 전체가 30개인 Dataset이였기 때문에 test로 인한 dataset은 10개이다
```

### xTrain값과 yTrain의 값으로 이루어진 산포도를 녹색으로 그려주기

```python
plt.scatter(xTrain,yTrain, c="g")
```

![image](https://user-images.githubusercontent.com/46669551/55393148-a4734e00-5577-11e9-928e-de8ffb4a33a7.png)

### xTrain에 대한 predict값을 선형 그래프로 출력해 주기

```python
plt.plot(xTrain, linearTest.predict(xTrain))
```

![image](https://user-images.githubusercontent.com/46669551/55393270-eac8ad00-5577-11e9-8499-bb88b34316ef.png)

### 두개의 그래프를 한번에 나타내기

```python
plt.scatter(xTrain,yTrain, c="g")
plt.plot(xTrain, linearTest.predict(xTrain))
plt.show()
```

![image](https://user-images.githubusercontent.com/46669551/55393368-151a6a80-5578-11e9-8812-cf433481c882.png)

### xTest값과 yTest의 값으로 이루어진 산포도를 빨강으로 그려주기

```python
plt.scatter(xTest,yTest, c="r")
```

![image](https://user-images.githubusercontent.com/46669551/55393932-23b55180-5579-11e9-9ef2-d9d2030d158f.png)

### xTest에 대한 predict값을 선형 그래프로 출력해 주기

```python
plt.plot(xTest, linearTest.predict(xTest))
```

![image](https://user-images.githubusercontent.com/46669551/55393992-47789780-5579-11e9-9cc9-2dd48bd996b4.png)

### 두개의 그래프를 한번에 나타내기

```python
plt.scatter(xTest,yTest, c="r")
plt.plot(xTest, linearTest.predict(xTest))
plt.show()
```

![image](https://user-images.githubusercontent.com/46669551/55394050-624b0c00-5579-11e9-809a-e8a85ac0ff73.png)

### 실제 데이터를 활용한 그래프 출력

```python
plt.scatter(x1,y1, c="b")
plt.plot(x1, linearTest.predict(x1))
plt.title("Experience/Salary")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
```

![image](https://user-images.githubusercontent.com/46669551/55394374-26647680-557a-11e9-9c4d-6f7046a30c3d.png)























```python


xTrain, xTest, yTrain, yTest = train_test_split(x1,y1,test_size=1/3,random_state=0)
linearTest = LinearRegression()
linearTest.fit(xTrain,yTrain)
yprediction = linearTest.predict(xTest)
print(yprediction)
plt.scatter(xTrain,yTrain, c="g")
plt.plot(xTrain, linearTest.predict(xTrain))
plt.show()

plt.scatter(xTest,yTest, c="r")
plt.plot(xTest, linearTest.predict(xTest))
plt.show()

type(data2["Salary"])
data2["Salary"].values
type(data2["Salary"].values)
plt.scatter(x,y, c="b")
plt.plot(x,y)
plt.title("Experience/Salary")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
```

