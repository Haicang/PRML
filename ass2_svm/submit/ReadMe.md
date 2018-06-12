# README

> 文件说明：
>
> 算法代码在 svm.py 中，utils.py 中有一些用于可视化的函数
>
> test_model.ipynb 是对不同模型的测试，show_model.ipynb 是可视化

There are 4 classes defined in  `svm.py` ：

- `SVM`  svm for 2-class
- `multiSVM`  svm for multi-class 
- `Linear`  Linear classifier based on Square Error
- `Logistic`  Logistic Regression

### class SVM

```python
class SVM(C=1, kernel='g', power=2, loss=None):
```

- C: penalty parameter C of the error term
- kernel: kernel functions, optional: 'g'—gaussian, 'p': polynomial, 'l': linear
- loss: if `None`, svm use convex optimization , if 'hinge'(kernel=='l' only) use hinge loss, else Exception raised
- power: the power of ploynomial kernel, and only used in it.

```python
def train(data_train, epochs=1000, lr=0.01, l=0.01, show=False):
```

- data_train: np.ndarray of shape(m, n + 1), the last colunm is the targets of training set, m is the number of training examples while n is the number of features
- epochs: used for linear kernel on hinge loss only
- The parameters behind `epochs` is only used in hinge loss
  - lr: learning rate
  - l: l-2 penalty
  - show: show the process of the gradient descent
- **return** None

```python
def fit(X, y, epochs=1000, lr=0.01, l=0.01, show=False)
```

Another version of `def train`

- X: np.ndarray of shape(m, n), training examples 
- y: np.ndarray of shape(n, ), training targets
- **return** None

```python
def predict(x):
```

- x: np.ndarray of shape(m, n)
- **return** y, predictions based on input x, np.ndarray of shape(m, )



### class multiSVM

```python
class multiSVM(C=1, n_classes=2, kernel='g', loss=None, decision_function_shape='ovr'):
```

multi-class SVM classifier based on multiple SVM. `C`  `kernel`  `loss`  have the same meaning

- n_classes: integer, the number of target kinds
- decision_function_shape: be either 'ovr'(one-versus-rest) or 'ovo'(one-versus-one), otherwise, Exception raised

```python
def train(data, epochs=100):
```

```python
def fit(X, t, epochs=100):
```

Similar to `def train` and `def fit` defined in `class SVM` 



### class Linear()

```python
class Linear():
```

Linear classifier base on (Mean) Squared Error. Bacuase MSE is not a suitable loss for classification, this classifier is **only useful to the homework**

```python
def fit(X, y, epochs=1000, lr=0.01, l=0.01, show_loss=False):
```

Train the classifier

- X: np.ndarray of shape(m, n)  training data
- y: np.ndarray of shape(m, ) training targets
- lr: learning rate
- l: l2-penalty
- show_loss: plot the loss during the training precess or not
- **return** None

```python
def train(data, epochs=1000, lr=0.01, l=0.01, show_loss=False)
```

Another version of `fit` data = [X, y]

```python
def predict(X):
```

Give predictions on test example X

- X: np.ndarray of shape(m, n)
- **return** y:  np.ndarray of shape(m, ), the predictions



### class Logistic()

Logistic Regression which has the same API as `Linear()`

