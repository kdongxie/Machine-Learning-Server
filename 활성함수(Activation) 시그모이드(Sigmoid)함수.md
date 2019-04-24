# 활성함수(Activation) 시그모이드(Sigmoid)함수
### 로지스틱 회귀분석 또는 Neural network의 Binary classification 마지막 레이어의 활성함수로 사용하는 시그모이드

# ![image](https://user-images.githubusercontent.com/46669551/55845446-d2125580-5b7c-11e9-92ee-6e4ecd1a689b.png)
### 에 대해 살펴보겠다. 다음 그림은 시그모이드 함수의 그래프다.

# ![image](https://user-images.githubusercontent.com/46669551/55845471-ee15f700-5b7c-11e9-9c11-3cb3a52d1d42.png)
### 데이터를 두 개의 그룹으로 분류하는 문제에서 가장 기본적인 방법은 로지스틱 회귀분석이다. 회귀분석과의 차이는 회귀분석에서는 우리가 원하는 것이 예측값(실수)이기 때문에 종속변수의 범위가 실수이지만 로지스틱 회귀분석에서는 종속변수  y 값이 0 또는 1을 갖는다. 그래서 우리는 주어진 데이터를 분류할 때 0인지 1인지 예측하는 모델을 만들어야 한다.

### 0을 실패, 1을 성공 이라고 하겠다.

### 로지스틱 회귀분석에서 데이터를 두 개의 그룹으로 분리하는데 선형함수(직선)을 사용하면 안되는 이유를 먼저 살펴보겠다. 예를 들어서 다음과 같이 데이터가 주어졌다고 가정해보자.

# ![image](https://user-images.githubusercontent.com/46669551/55845738-076b7300-5b7e-11e9-8257-429cbd4c00c2.png)

### 먼저 데이터를 다음과 같이 선형함수로 분류를 했다고 하자.

# ![image](https://user-images.githubusercontent.com/46669551/55845758-1eaa6080-5b7e-11e9-91a1-be293d81f2aa.png)

### 즉, 함수값이  12 이 나오는  x 를 기준으로 성공과 실패가 구분된다고 보면 된다. 우선 현재의 데이터와 결과를 봤을 때 잘 분한 것 처럼 보인다. 

### 하지만  x=20 이라는 새로운 데이터가 성공인지 실패인지 알아보기 위해 선형함수에 적용해보면 함숫값은 1보다 큰 값이 되고 이 함숫값에 적절한 의미를 부여하기 어렵다. 

### 또 다른 이유로, 학습데이터에 새로운 값(20,1)이 추가된다면( x=20 을 갖고 성공한 경우) 두 그룹(성공과 실패)으로 분류하는 직선은 다음과 같이 변경될 것이다.

# ![image](https://user-images.githubusercontent.com/46669551/55845819-5c0eee00-5b7e-11e9-9ac3-cbceaf8087cc.png)

### 다른  x 값에 비해 큰 x=20 으로 인해서 선형함수의 기울기가 더 작아지고 새로운 데이터의 추가로 인해서 기존에 잘 분류되었던 (9,1)과 (10,1)을 분류하는데 실패하게 된다. 

### 즉, 새로운 데이터의 추가가 기존의 분류 모델에 큰 영향을 미치게 된다.

### 그래서 로지스틱 회귀분석에서는 다음과 같은 형태의 함수를 활성함수로 사용하여 데이터를 성공과 실패로 분류한다.

# ![image](https://user-images.githubusercontent.com/46669551/55845924-be67ee80-5b7e-11e9-99ce-e1a55cacc43a.png)

### 함수의 특징을 살펴보면 다음과 같다.

### 성공과 실패를 구분하는 부분은 경사가 급하고 나머지 부분에서는 경사가 완만하다.

### y=1 , y=0  두 평행선이 점근선이고 치역은 (0,1)이다. 즉 위와 같은 활성함수의 함숫값은 성공확률이라는 의미로 해석할 수 있다(1을 실패라고 정의했다면 실패 확률로 해석하면 된다).

### 이러한 특징을 만족하는 함수로 시그모이드  s(z)=11+e−z  함수를 활성함수로 사용한다. 여기서 활성함수를 시그모이드를 사용하는 것이 맞는지에 대한 질문에 대답을 해야한다. 

### 혹자는 시그모이드가 계단함수(Step function)의 미분가능한 형태이기 때문이라고 하고 또는 작은 자극에는 감각을 거의 느끼지 못하다가 어떤 임계값을 넘어가야 감각을 느끼는 우리의 신경망 세포와 비슷하기 때문이라고 말하지만 이것은 수학적이지 못하다.

### 그래서 왜 시그모이드를 사용할까?

### 단순선형회귀분석(독립변수의 갯수가 1개)은 앞에서 언급한 것처럼 목표가 실수값 예측이기 때문에 선형함수  y=wx+b 를 이용하여 예측한다(예측 변수의 수가 하나인 경우). 

### 하지만 로지스틱 회귀분석에서는 종속변수가 0 또는 1이기 때문에  y=wx+b 을 이용해서 예측하는 것은 의미가 없다고 앞에서 살펴보았다. 

### 그래서 Odds 를 이용하는데 Odds 는 다음과 같이 정의 된다. 

### 확률 p 가 주어져 있을 때

# ![image](https://user-images.githubusercontent.com/46669551/55846708-a47bdb00-5b81-11e9-95af-1ade22bedab4.png)
### 로 정의한다. 확률  p 의 범위가 (0,1)이라면 Odds(p) 의 범위는 (0,∞ )이 된다.

### Odds 에 로그함수를 취한  log(Odds(p)) 은 범위가 (-∞ ,∞ )이 된다. 

### 즉, 범위가 실수 전체이다.

### log(Odds(p)) 의 범위가 실수이므로 이 값에 대한 선형회귀분석을 하는 것은 의미가 있다.

# ![image](https://user-images.githubusercontent.com/46669551/55846782-e442c280-5b81-11e9-9561-fff7cbd3b091.png)

### 으로 선형회귀분석을 실시해서  w 와 b 를 얻을 수 있다. 

### 위 식을 p 로 정리하면 다음과 같은 식을 얻을 수 있는데 이 식이 시그모이드이다.

# ![image](https://user-images.githubusercontent.com/46669551/55846806-fde40a00-5b81-11e9-9252-e267c315a8ab.png)

### x  데이터가 주어졌을 때 성공확률을 예측하는 로지스틱 회귀분석은 학습데이터를 잘 설명하는 시그모이드 함수의  w 와 b 를 찾는 문제다.

# ![image](https://user-images.githubusercontent.com/46669551/55847193-5a93f480-5b83-11e9-863c-6b577bd91841.png)

## 예제

```python
import tensorflow as tf
# Logistic regression
# Decision Boundary 를 만들어 주는 형태 
xdata = [
    [1,2],
    [2,1],
    [2,3],
    [4,1],
    [5,1],
    [5,4]
]
ydata = [
    [0],
    [0],
    [1],
    [1],
    [0],
    [1]
]
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
#x, y => 0 or 1
```

```python
w = tf.Variable(tf.random_normal([2,1]),name = "weight")
b = tf.Variable(tf.random_normal([1]), name = 'bias')
```

```python
# Hypothesis sigmoid
hypothesis = tf.sigmoid(tf.matmul(X,w) + b) 
# matmul : metrix를 multiply 해주는 함수
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y)* tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
```

```python
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# cast 함수
# hypothesis 의 값이 > 0.5 => 1 , hypothesis 의 값이 < 0.5 ==> 0 
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype= tf.float32))
```

```python
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 변수 초기화
```

```python
for i in range(2000):
    cost_val,_ =sess.run([cost,train], feed_dict={X:xdata, Y:ydata})
    if i % 200 ==0:
        print(i, 'Cost 값 :',cost_val)
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                      feed_dict={X:xdata, Y:ydata})
    print('sigmoid 함수 값:',h, 'regression 값:',c, '정확도: ',a)
```

```python
0 Cost 값 : 0.47227108
sigmoid 함수 값: [[0.44176015]
 [0.27903938]
 [0.7280965 ]
 [0.39044702]
 [0.4517668 ]
 [0.937485  ]] regression 값: [[0.]
 [0.]
 [1.]
 [0.]
 [0.]
 [1.]] 정확도:  0.8333333
sigmoid 함수 값: [[0.44173115]
 [0.27900726]
 [0.72810525]
 [0.3904375 ]
 [0.45177174]
 [0.93750393]] regression 값: [[0.]
 [0.]
 [1.]
 [0.]
 [0.]
 [1.]] 정확도:  0.8333333

            .
            .
            .
```





