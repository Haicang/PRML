
# Report

<center>周海沧  15307130269</center>


```python
import numpy as np
import matplotlib.pyplot as plt

from gm import GaussianMixture, generate_data
```

## 1. 数据的生成

在 generate_data 中我将三个二维正态的随机数放在同一个数组里，然后对这个数组进行 random shuffle，就得到了训练数据。原始的训练数据展示如下：


```python
data = generate_data()
plt.figure(1)
plt.scatter(data[:, 0], data[:, 1])
plt.show()
```



![png](output_5_1.png)


注：验证矩阵是否 PSD 可以用检验所有特征值都大于等于 0 的方法，用这种方式检验，我所选的 cov 矩阵都是 PSD 的；stackoverflow 上的解释是，浮点数精度的问题，这似乎是 numpy 的一个 bug

## 2. 模型

使用 EM 算法对 高斯混合模型 进行优化，相关的公式在书上和讲义上都有了。对于训练的初始化，可以用一个 kmeans。训练和结果展示如下。


```python
model = GaussianMixture(n_cluster=3)
model.fit(data, init='kmeans', init_iter=10)
```


```python
c = model.predict(data)
```


```python
plt.figure(3)
plt.scatter(data[:, 0], data[:, 1], c=c)
plt.show()
```


![png](output_11_0.png)

