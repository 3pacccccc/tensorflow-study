import numpy
import matplotlib.pyplot as plt
import tensorflow


"""
用梯度下降的优化方法来快速解决线性回归问题
"""

# 构建数据
points_num = 100
vectors = []
# 用numpy的正太随机分布函数生成100个点
# 这些点的(x, y)坐标值对应线性方程y = 0.1 * x + 0.2
# 权重(Weight)0.1， 偏差(bias)0.2
for i in range(points_num):
    x1 = numpy.random.normal(0.0, 0.66)  # x1取均值为0，标准差为0.66的正太分布随机数
    y1 = 0.1 * x1 + 0.2 + numpy.random.normal(0.0, 0.04)
    vectors.append([x1, y1])

x_data = [v[0] for v in vectors]    # 真实的x的坐标
y_data = [v[1] for v in vectors]    # 真实的y的坐标

plt.plot(x_data, y_data, 'r.', label="original data")   # r.表示红色.形状的点
plt.title("Linear Regression using Gradient Descent")
plt.legend()    # 调用之后会展示label="original data"标签
plt.show()


# 构建线性回归值

# 初始化weight,参数含义： shape=[1], min=-1, max=1
W = tensorflow.Variable(tensorflow.random_uniform([1], -1, 1))
# 初始化bias
b = tensorflow.Variable(tensorflow.zeros([1]))

y = W * x_data + b

# 定义loss function(损失函数) 或cost function(代价函数)
# 对tensor的所有唯独计算((y - y_data)^2)之和/N
loss = tensorflow.reduce_mean(tensorflow.square(y - y_data))

# 用梯度下降的优化器来优化我们的“lost function”
optimizer = tensorflow.train.GradientDescentOptimizer(0.5)    # 设置学习率0.5
train = optimizer.minimize(loss)

# 创建回话
sess = tensorflow.Session()

# 初始化数据流图中的所有变量
init = tensorflow.global_variables_initializer()
sess.run(init)

# 训练20步
for step in range(20):
    sess.run(train)   # 优化每一步
    # 打印每一步的损失， 权重和偏差
    print("step=%d, loss=%f, [Weight=%f Bias=%f" % (step, sess.run(loss), sess.run(W), sess.run(b)))


# 图像2：
plt.plot(x_data, y_data, 'r.', label="original data")   # r.表示红色.形状的点
plt.title("Linear Regression using Gradient Descent")
plt.plot(x_data, sess.run(W)*x_data+sess.run(b), label="fitted line")
plt.legend()    # 调用之后会展示label="original data"标签
plt.xlabel('x轴')
plt.ylabel('y轴')
plt.show()

sess.close()
