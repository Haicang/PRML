"""
This script is used for debug and report.ipynb
"""


from base import *
import numpy as np
import matplotlib.pyplot as plt
from linear_reg import load_data, evaluate


def test(model, lr, epochs, n=2, l2=0):
    
    assert type(model) is str
    
    train_file = 'train.txt'
    test_file = 'test.txt'

    # 载入数据
    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)
    print(x_train.shape)
    print(x_test.shape)

    # 使用线性回归训练模型，返回一个函数f()使得y = f(x)
#     f = main(x_train, y_train)
    if model == 'power':
        f = power(x_train, y_train, n=n, epochs=epochs, Print=False, learning_rate=lr, l2=l2)
    elif model == 'gaussian':
        f = gaussian(x_train, y_train, n=n, epochs=epochs, Print=False, learning_rate=lr, l2=0)
    elif model == 'sigmoid':
        f = sigmoid_base(x_train, y_train, n=n, epochs=epochs, Print=False, learning_rate=lr, l2=0)
    elif model == 'mix':
        f = mix(x_train, y_train, epochs=epochs, Print=False, learning_rate=lr, l2=0)
    elif model == 'jin':
        f = linear_regression(x_train, y_train, epochs=epochs, learning_rate=lr, poly=n, l2=l2)
    else:
        raise Exception

    # 计算预测的输出值
    y_test_pred = f(x_test)

    # 使用测试集评估模型
    std = evaluate(y_test, y_test_pred)
    print('预测值与真实值的标准差：{:.1f}'.format(std))

    # 显示结果
    plt.plot(x_train, y_train, 'ro', markersize=3)
    plt.plot(x_test, y_test, 'k')
    plt.plot(x_test, y_test_pred)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend(['train', 'test', 'pred'])
    plt.show()


if __name__ == '__main__':
    test('gaussian', lr=1e-7, n=3, epochs=50000, l2=0)