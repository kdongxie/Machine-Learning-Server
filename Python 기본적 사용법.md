# Python 기본적 사용법

## 기본 Python 제공 기능

### 1개의  변수 선언

```python
test = 1
print(test)
```

```
1
```

### 2개의  변수 선언

```python
a,b=1,2
print(a)
print(b)
print(a,b)
```

```
1
2
1 2
```

### 문자열 선언

```python
test2 = "test String"
print(test2)
```

```
test String
```

### 번언된 변수의 원소를 Slicing 하기

```python
print(test2[:1])
print(test2[0])
print(test2[-1:])
print(test2[:])
```

```
t
t
g
test String
```

### List type으로 선언하고 Slicing하기

```python
test3 = ['a','b','c','d']
print(test3[0])
print(test3[:])
print(test3[:3])
```

```
a
['a', 'b', 'c', 'd']
['a', 'b', 'c']
```

### 반복출력

```python
print(test3[1]*2)
print(test3[:3]*3)
```

```
bb
['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']
```

### 선택된 인덱스의 리스트의 값을 변환해주기

```python
test3[0] = '6'
print(test3)
```

```
['6', 'b', 'c', 'd']
```

### 복합된 type의 원소를 같는 변수를 선언하고 출력해주기

```python
test4 = (1,'a','c')
print(test4)
print(test4[0])
#test4[0] = [('a','b'),('g','e')]
test5 = [('a','b'),('g','e')]
print(test5)
test6 = ((1,2),('a','d'))
print(test6)
print(test6[0])
```

```
(1, 'a', 'c')
1
[('a', 'b'), ('g', 'e')]
((1, 2), ('a', 'd'))
(1, 2)
```

### Dictionary 선언하고 원하는 값을 출력하기

```python
test7 = {"home":'yarn',"age":30}
print(test7)
print(test7['home'])
print(test7['age'])
test7['age'] = 20
print(test7)
```

```
{'home': 'yarn', 'age': 30}
yarn
30
{'home': 'yarn', 'age': 20}
```

### 데이터가 없는 Dictionary를 선언하고 이후 데이터 넣기

```python
test8 = {}
print(test8)
test8["age"] = 20
print(test8)
test8[10] = "man"
print(test8)
print(test8.keys())
print(test8.values())
```

```
{}
{'age': 20}
{'age': 20, 10: 'man'}
dict_keys(['age', 10])
dict_values([20, 'man'])
```

### 데이터 삭제하기

```python
del test8
```

### Type강제변환하기

```python
test9 = '1'
print(type(test9))
test10 = int(test9)
print(type(test10))
```

```
<class 'str'>
<class 'int'>
```

### 조건문 if

```python
test11 = [1,2,3,4,5,6,7]
if(3 in test11):
    print('ok')
```

```
ok
```

### 조건문 if , else

```python
a,b,c = 1,2,1
if(a is b):
    print("ok")
else:
    print("no")
```

```
no
```

### 조건문 한줄에 선언해주기

```python
d = 10
if(d==10):print('ok')
```

```
ok
```

### 반복문 While

```python
cnt =0
while(cnt <10):
    print(cnt)
    cnt=cnt+1
```

```
0
1
2
3
4
5
6
7
8
9
```

### 반복문 While & else

```python
while(cnt <10):
    print(cnt)
    cnt=cnt+1
else:
    print('no')
```

```
no
```

### 무한 루프 While

```python
check =1
while(check):print('ok')
```

```
ok
ok
ok
ok
ok
ok
ok
ok
ok
ok
ok
ok
ok
ok
# 무한 루프가 돌게 된다.
```

### String type의 List를 받아 For문을 돌리기

```python
test12 = ['apple','samsung','google']
for i in range(len(test12)):
    print(test12[i])
```

```
apple
samsung
google
```

### For문을 선언시켜주고 실행시키지 않을 때 쓰는 방법 pass

```python
for i in range(len(test12)):
    pass
```

```python
# 처리는 되지 않으나 오류가 생기지 않음. 알고 있음 좋다. 
```

### Function 선언해주기 def

```python
def add(a,b):
    return a+b
test13 = add(1,7)
print(test13)
```

```
8
```

### 전역 Function을 선언하고 return값으로 받아주기

```python
def testfunc(d):
    d = [10,20,30]
    print(d)
    return
```

```python
mylist= [1,2,3]
testfunc(mylist)
print(mylist)
```

```
[10, 20, 30]
[1, 2, 3]
```

### 숫자를 받는 Function을 선언하기

```python
def add(a=0,b=0,c=0):
    return print(a,b,c)
```

```python
add(a=1,b=2,c=5)
```

```
1 2 5
```

```python
add(1,2,3)
```

```
1 2 3
```

```python
add(b=2)
```

```
0 2 0
```

## Numpy

### Numpy 를 이용하여 Array를 선언하기

```python
import numpy as np
test = np.array([1,2,3])
print(test)
test2 = np.array([[4,5,6],[7,8,9]])
print(test2)
test3 = np.array([1,2,3,4,5,6,7,8],ndmin = 2)
print(test3)
```

```
[1 2 3]
[[4 5 6]
 [7 8 9]]
[[1 2 3 4 5 6 7 8]]
```

### Array의 형태를 알아보는 shape

```python
print(test.shape)
print(test2.shape)
print(test3.shape)
```

```
(3,)
(2, 3)
(1, 8)
```

### 2차원의 향렬을 선언하는 방법

```python
test4 = np.array([[4,5,6],[7,8,9]])
print(test4)
test4.shape
```

```
[[4 5 6]
 [7 8 9]]
(2, 3)
```

### 차원을 바꾸어 주는 .reshape

```python
test5 = test4.reshape(3,2)
print(test5)
```

```
[[4 5]
 [6 7]
 [8 9]]
```

### np.arange를 사용한 list를 slicing하기

```python
test6 = np.arange(10)
print(test6)
print(test6[0])
print(test6[:5])
print(test6[-1])
print(test6[1:-1])
print(test6[1:])
print(test6.shape)
```

```
[0 1 2 3 4 5 6 7 8 9]
0
[0 1 2 3 4]
9
[1 2 3 4 5 6 7 8]
[1 2 3 4 5 6 7 8 9]
(10,)
```

```python
test7 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(test7.shape)
print(test7[1])
print(test7[0])
print(test7[2])
print(test7[1:])
print(test7[1,1])
print(test7[2,2])
print(test7[:,1])
print(test7[1,:])
print(test7[:,[0,2]])
print(test7.reshape(1,9))
```

```
(3, 3)
[4 5 6]
[1 2 3]
[7 8 9]
[[4 5 6]
 [7 8 9]]
5
9
[2 5 8]
[4 5 6]
[[1 3]
 [4 6]
 [7 9]]
[[1 2 3 4 5 6 7 8 9]]
```

### 최소 값과 최대 값 & 행렬안의 최소 값, 최대 값을 구하기

```python
test8 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(np.amin(test8)) # 최소값
print(np.amax(test8)) # 최대값
print(np.amin(test8,1)) # column 기준 원소들중의 최소 값
print(np.amin(test8,0)) # row 기준 원소들 중 최소 값
print(np.amax(test8,1))
print(np.amax(test8,0))
```

```
1
9
[1 4 7]
[1 2 3]
[3 6 9]
[7 8 9]
```

### 최소 값 과 최대 값의 범위를 나타내 주는 np.ptp

```python
np.ptp(test8) # 최소값과 최대값의 범위를 나타내준다 (min ~ max)
```

```
8
```

### 전체의 원소 값들 중 50%에 해당되는 값을 보고 싶을 때

```python
np.percentile(test8,50) # 50%기준으로 보고싶을때
```

```
5.0
```

```python
np.percentile(test8,50,1) # column 기준 중간
```

```
array([2., 5., 8.])
```

```python
np.percentile(test8,50,0) # row 기준 중간
```

```
array([4., 5., 6.])
```

### 중간 값, 중앙 값, 평균, 표준편차, 분산

```python
t = [1,1,1,1,3,4,4,5,7,7,6,7,10,29,45,100]
np.median(t) # 중앙값
```

```
5.5
```

```python
np.mean(t) # 평균
```

```
14.4375
```

```python
np.average(t) # 평균
```

```
14.4375
```

```python
np.std(t) #표준편차
```

```
24.854498461043224
```

```python
data = [1,2,3,4]
print(np.mean(data)) # 평균
print(np.std(data)) # 표준편차
print(np.var(data)) # 분산
```

```
2.5
1.118033988749895
1.25
```

## statistics

### statistics 를 사용한 함수

```python
import statistics
test_data = [5,2,5,6,1,1,2,5,7,8]
print(statistics.mean(test_data))
print(statistics.median(test_data))
print(statistics.stdev(test_data))
print(statistics.variance(test_data))
print(statistics.mode(test_data)) # 최빈 값
#https://docs.python.org/3/library/statistics.html
```

```
4.2
5.0
2.5298221281347035
6.4
5
```

## matplotlib 

### matplotlib을 활용한 그래프 출력

```python
import matplotlib.pyplot as plt
test_data2 = [5,2,5,6,1,1,2,5,7,8]
plt.hist(test_data2, bins=10)
plt.show() # 빈도 수를 히스토 그램으로 출력
print("중간 값", np.median(test_data2))
print("최빈 값", statistics.mode(test_data2))
print("최대 값", np.amax(test_data2))
```

![image](https://user-images.githubusercontent.com/46669551/55391304-15186b80-5574-11e9-94d6-65bd5f0dd391.png)

