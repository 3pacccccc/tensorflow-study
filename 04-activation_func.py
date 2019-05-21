import tensorflow
import matplotlib.pyplot as plt
import numpy

"""
使用TensorFlow的激活函数结合numpy绘制出非线性的函数图像
"""

x = numpy.linspace(-7, 7, 180)  # 在-7到7之间去180个点


# 激活函数的原始实现
def sigmoid(inputs):
    y = [1 / float(1 + numpy.exp(-x)) for x in inputs]  # y = 1/e^(-x)
    return y


def relu(inputs):
    y = [x * (x > 0) for x in inputs]
    return y


def tanh(inputs):
    y = [(numpy.exp(x) - numpy.exp(-x)) / float(numpy.exp(x) + numpy.exp(-x)) for x in inputs]
    return y

def softplus(inputs):
    y = [numpy.log(1 + numpy.exp(x)) for x in inputs]
    return y

# 经过TensorFlow的激活函数处理的各个Y值
y_sigmoid = tensorflow.nn.sigmoid(x)
y_relu = tensorflow.nn.relu(x)
y_tanh = tensorflow.nn.tanh(x)
y_softplus = tensorflow.nn.softplus(x)

# 创建回话
sess = tensorflow.Session()

# 运行
y_sigmoid, y_relu, y_tanh, y_softplus = sess.run([y_sigmoid, y_relu, y_tanh, y_softplus])

# 创建各个激活函数的图像
plt.subplot(221)
plt.plot(x, y_sigmoid, c='red', label="sigmoid")
plt.ylim(-0.2, 1.2)
plt.legend(loc='best')   # loc='best'使label标签放在最合适的位置

plt.subplot(222)
plt.plot(x, y_sigmoid, c='red', label="sigmoid")
plt.ylim(-1, 6)
plt.legend(loc='best')   # loc='best'使label标签放在最合适的位置

plt.subplot(223)
plt.plot(x, y_tanh, c='red', label="tanh")
plt.ylim(-1.3, 1.3)
plt.legend(loc='best')    # loc='best'使label标签放在最合适的位置

plt.subplot(224)
plt.plot(x, y_softplus, c='red', label="softplus")
plt.ylim(-1, 6)
plt.legend(loc='best')    # loc='best'使label标签放在最合适的位置

# 显示图像
plt.show()

# 关闭会话
sess.close()