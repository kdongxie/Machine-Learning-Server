#  Anaconda with Tensorflow in JupyterNotebook

## Tensorflow의 기본 구성

![image](https://user-images.githubusercontent.com/46669551/55773040-dc274c00-5ac9-11e9-92e3-ced655425fed.png)

### Tensor Data Structure

- Rank
  - 텐서 (Tensor) (Tensor) (Tensor) 내에 기술된 차원의 단위는 랭크 (rank) 라고 부름 .
  - 텐서의 차원 수를 나타냄 .
  - 텐서의 랭크는 텐서의 차수 또는 n-차원으로 정의 될 수 있음 .
- Shape Shape Shape
  - 행과 열의 수는 함께 Tensor Tensor Tensor 의 모양을 정의함 .
- Type Type
  - 유형은 Tensor Tensor Tensor 의 요소에 지정된 데이터 유형을 나타냄 .

### TensorFlow - Basics

- TensorFlow의 다양한 함수
  - TensorFlow는 다양한 차원을 포함함
    - 1차원 텐서
    - 2차원 텐서
    - Tensor Handling & Manipulations

## 설치

```
# Anaconda Prompt ==>
# activate tensorflow
# conda create --name tensorflow python=3.7
```

## 실행

```
activate tensorflow
```

```
(tensorflow) C:\Users\Administrator>
```

## Tensorflow 설치

```
pip install tensorflow
```

```c
Successfully built termcolor absl-py gast
Installing collected packages: numpy, six, keras-preprocessing, protobuf, werkzeug, markdown, absl-py, grpcio, tensorboard, astor, termcolor, gast, h5py, keras-applications, pbr, mock, tensorflow-estimator, tensorflow
Successfully installed absl-py-0.7.1 astor-0.7.1 gast-0.2.2 grpcio-1.19.0 h5py-2.9.0 keras-applications-1.0.7 keras-preprocessing-1.0.9 markdown-3.1 mock-2.0.0 numpy-1.16.2 pbr-5.1.3 protobuf-3.7.1 six-1.12.0 tensorboard-1.13.1 tensorflow-1.13.1 tensorflow-estimator-1.13.0 termcolor-1.1.0 werkzeug-0.15.2

(tensorflow) C:\Users\Administrator>
```

> GPU를 사용하려면 `pip install tensorflow -gpu` 로 

## Jupyter Notebook 설치

```python
# Anaconda command prompt ==>
pip install jupyter
```



## Jupyter Notebook - Python3 수행

```
# Anaconda prompt 실행
activate tensorflow
jupyter notebook
# jupyter notebook 실행 여부 확인
```

## TensorFlow 수행 및 운용 검사

```python
import tensorflow as tf #문제가 없으면 OK
```

```python
hello = tf.constant('Hello, Tensorflow') # constant 구성하기
```

```python
sess = tf.Session()
```

```python
print(sess.run(hello))
```

```python
# 출력 값
b'Hello, Tensorflow'
```

### Tensorflow 사용 예제

```python
x1 = tf.constant([1,2,3,4])
y1 = tf.constant([5,6,7,8])
```

```python
result = tf.multiply(x1,y1)
```

```python
print(result)
```

```python
# 출력 값
Tensor("Mul:0", shape=(4,), dtype=int32)
```

```python
sess = tf.Session()
sess.run(result) # Session.run()을 구동시켜야 값이 출력이 된다.
```

```python
# 출력 값
array([ 5, 12, 21, 32])
```

### Numpy사용 가능

```python
import numpy as np
test1 = np.array([1.5,1,5.0,23.0])
test1
```

```python
# 출력 값
array([ 1.5,  1. ,  5. , 23. ])
```

### Tensorflow에서 기본적으로 matplotlib 지원이 되지 않음 따라서  설치해야함

```python
!pip install matplotlib
```

```python
# tensorflow codes
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
```

### Tensorflow의 기본 함수 

```python
# tensorflow codes
a = tf.constant(5.0) # Tensor 1
b = tf.constant(6.0) # Tensor 2
c = a*b # Tensor 3 (Multiply 가능 Tensor)
d = a/b
```

```python
c # Tensor에 정의된 변수임 
```

```python
# 출력 값
<tf.Tensor 'mul:0' shape=() dtype=float32>
```

```python
# computational graph
sess = tf.Session() # Session을 통해 Tensor의 값을 출력해준다. 
output_c = sess.run(c)
```

```python
output_c
```

```python
# 출력 값
30.0
```

```python
sess.run(a), sess.run(b), sess.run(c), sess.run(d)
```

```python
# 출력 값
(5.0, 6.0, 30.0, 0.8333333)
```

### PlaceHolder (수행 할때에 Parameter를 넣어주어야 수행이  된다.)

```python
# placeholder
a1 = tf.placeholder(tf.float32)
a1
```

```python
# 출력 값
<tf.Tensor 'Placeholder_1:0' shape=<unknown> dtype=float32>
```

```python
a1 = tf.placeholder(tf.float32) # 명시만 함
b1 = tf.placeholder(tf.float32) # 명시만 함
c1 = a1 * b1 # 함수 Tesor
output_c1 = sess.run(c1, {a1:[1,2],b1:[2,4]}) # Parameter(값) 을 넣어준다
output_c1
```

```python
# 출력 값
array([2., 8.], dtype=float32)
```

### Variables

```python
# variables
d1 = tf.Variable([0.8], dtype=tf.float32)
```

```python
# 출력 값
WARNING:tensorflow:From C:\Users\Administrator\Anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
```

### Linear Regression

```python
import tensorflow as tf
```

```python
# Linear Regression -> Y = b + aX, b=> Y-intercept
a = tf.Variable([.4],dtype=tf.float32)
b = tf.Variable([-.4], dtype= tf.float32)
```

```python
X = tf.placeholder(tf.float32)
```

```python
linear_model = a * X + b
```

```python
sess = tf.Session()
init = tf.global_variables_initializer() # 변수 초기화
```

```python
sess.run(init) # Session 수행
```

```python
sess.run(linear_model, {X:[1,2,3,4]}) # Parameter 입력
```

```python
# 출력 값
array([0.8, 1.2, 1.6, 2. ], dtype=float32)
```

### TensorBoard

#### Jupyter Notebook의 기본 루트에 log 파일 만들기

``` python
import tensorflow as tf
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a*b
sess = tf.Session()
writer = tf.summary.FileWriter('./logs',sess.graph)
sess.run(c)
```

```python
# Linear Regression -> Y = b + aX, b=> Y-intercept
a = tf.Variable([.4],dtype=tf.float32)
b = tf.Variable([-.4], dtype= tf.float32)
X = tf.placeholder(tf.float32)
linear_model = a * X + b
sess = tf.Session()
init = tf.global_variables_initializer() # 변수 초기화
sess.run(init)
sess.run(linear_model, {X:[1,2,3,4]})
```



#### 루트에서 확인하기

![image](https://user-images.githubusercontent.com/46669551/55775598-c1a6a000-5ad4-11e9-87b5-7f3875f96b80.png)

> Event Log 가 생성된을 볼 수 있다.

#### Tensor Board 실행하기

![image](https://user-images.githubusercontent.com/46669551/55775565-acca0c80-5ad4-11e9-8ffa-3e8a55c928e9.png)

1. Anaconda Prompt 실행
2. TensorFlow 활성화 시키기 : `activate tensorflow`
3. Tensor Board 실행시키기 : `tensorboard --logdir=C:\Pythontest3\logs` <- log폴더가 생성된 루트를 입력
4. 크롬 또는 Internet Explore 를 통해 접속 : `http://localhost:6006`

#### Tensor Board 확인하기

![image](https://user-images.githubusercontent.com/46669551/55775721-2c57db80-5ad5-11e9-8e90-ca18e2f6286b.png)

## 예제

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

```python
writer = tf.summary.FileWriter('./logs',sess.graph)
# 임의의 그래프를 만들기
trainx = np.linspace(-1,1,101)
trainy = 3 * trainx + np.random.randn(*trainx.shape)*0.33 
```

```python
plt.scatter(trainx, trainy)
plt.show()
```

![image](https://user-images.githubusercontent.com/46669551/55779874-99716e00-5ae1-11e9-9cc8-776e7dd9a4b2.png)

```python
X = tf.placeholder("float")
Y = tf.placeholder(tf.float32) # 두개 모두 똑같은 명령어
```

```python
w = tf.Variable(0.0,name = 'weights')
init = tf.global_variables_initializer() # 전체 변수를 초기화 한 후 시작하겠다.
```

```python
tf.__version__ # TensorFlow version 
```

```python
#출력 값
'1.13.1'
```

```python
y_mode = tf.multiply(X,w)
cost = (tf.pow(Y-y_mode,2)) # Model을 넣어주기 # pow함수 3제곱 함수
```

```python
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
```

```python
sess = tf.Session()
sess.run(init)
for i in range(100):
    for(x,y) in zip(trainx, trainy):
        sess.run(train_op, feed_dict={X:x,Y:y})
```

```python
print(sess.run(w))
```

```python
# 출력 값
3.0404713
```

#### Tensor Board 확인

![image](https://user-images.githubusercontent.com/46669551/55780076-0c7ae480-5ae2-11e9-9478-2ff386130139.png)

![image](https://user-images.githubusercontent.com/46669551/55780624-5adcb300-5ae3-11e9-96dd-b008503c3f93.png)

![image](https://user-images.githubusercontent.com/46669551/55780657-6d56ec80-5ae3-11e9-9086-5f4317ad3f51.png)