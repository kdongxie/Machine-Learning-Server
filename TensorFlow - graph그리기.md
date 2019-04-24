

# TensorFlow 

## Tensor Board - graph 확인하기

```python
import tensorflow as tf
```

```python
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(5.0, tf.float32)
node3 = tf.add(node1, node2)

sess = tf.Session()
writer = tf.summary.FileWriter('./logs',sess.graph)
```

```python
sess.run(node3)
```

```python
# 출력 값
8.0
```

```python
a1 = tf.placeholder(tf.float32)
b1 = tf.placeholder(tf.float32)
adder_node = a1 + b1

sess.run(adder_node, feed_dict = {a1:3, b1:5})

sess.run(adder_node, feed_dict= {a1:[3,2], b1:[5,4]})
writer = tf.summary.FileWriter('./logs',sess.graph)
```

```python
addTest = adder_node * 3
```

```python
sess.run(addTest, feed_dict = {a1:4, b1:5})
```

```python
# 출력 값
27.0
```

```python
writer = tf.summary.FileWriter('./logs',sess.graph)
```

### TensorBoard

![image](https://user-images.githubusercontent.com/46669551/55844318-accf1880-5b77-11e9-80c0-91828e049cdd.png)

