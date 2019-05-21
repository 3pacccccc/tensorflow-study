import numpy
import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

# 下载并载入MNIST手写数字库(55000 * 28 * 28) 55000张训练图像
mnist = input_data.read_data_sets('mnist_data', one_hot=True, )    # one_hot 独热码的编码(encoding)形式

# None表示张量(Tensor)的第一个维度可以是任何长度
# 28*28*1表示一张28*28大小，只有1个颜色图片
input_x = tensorflow.placeholder(tensorflow.float32, [None, 28 * 28]) / 255.  # 除以255是因为是0-255灰度值的范围
output_y = tensorflow.placeholder(tensorflow.int32, [None, 10])   # 输入10个数字的预测
input_x_images = tensorflow.reshape(input_x, [-1, 28, 28, 1])   # 改变形状之后的输入

# 从Test（测试）数据集离选取3000个手写数字的图片和对应标签
test_x = mnist.test.images[:3000]  # 图片
test_y = mnist.test.labels[:3000]  # 标签

# 构建卷积神经网络
# 第一层卷积
conv1 = tensorflow.layers.conv2d(
    inputs=input_x_images,      # 形状[28, 28, 1]
    filters=32,                 # 32个过滤器，输出的深度(depth)是32
    kernel_size=[5, 5],         # 过滤器的二维大小是5*5
    strides=1,                  # 过滤器的步长是1
    padding='same',             # same表示输出的大小不变， 因此需要在外围补0两圈
    activation=tensorflow.nn.relu   # 激活函数的Relu
)

# 第一层池化(压采样)
pool1 = tensorflow.layers.max_pooling2d(
    inputs=conv1,               # 形状[28, 28, 32]
    pool_size=[2, 2],           # 过滤器的二维大小是[2, 2]
    strides=2,                  # 步长是2
)   # 形状是[14, 14, 32]

# 第二层卷积
conv2 = tensorflow.layers.conv2d(
    inputs=pool1,      # 形状是[14, 14, 32]
    filters=64,                 # 64个过滤器，输出的深度(depth)是64
    kernel_size=[5, 5],         # 过滤器的二维大小是5*5
    strides=1,                  # 过滤器的步长是1
    padding='same',             # same表示输出的大小不变， 因此需要在外围补0两圈
    activation=tensorflow.nn.relu   # 激活函数的Relu
)   # 形状是[14, 14, 64]

# 第二层池化(压采样)
pool2 = tensorflow.layers.max_pooling2d(
    inputs=conv2,               # 形状[14, 14, 64]
    pool_size=[2, 2],           # 过滤器的二维大小是[2, 2]
    strides=2,                  # 步长是2
)   # 形状是[7, 7, 64]

# 平坦化(flat)
flat = tensorflow.reshape(pool2, [-1, 7 * 7 * 64])    # 将pool2得到的形状改变为需要的形状，-1表示系统根据需要的形状自己推断在该方向的长度

# 1024个神经元的全连接层
dense = tensorflow.layers.dense(inputs=flat, units=1024, activation=tensorflow.nn.relu)

# Dropout: 丢弃50%， rate=0.5
dropout = tensorflow.layers.dropout(inputs=dense, rate=0.5)

# 10个神经元的全连接层， 这里不用激活函数来做非线性化了
logits = tensorflow.layers.dense(inputs=dropout, units=10)    # 输出。形状[1, 1, 10]

# 计算误差(计算Cross entropy(交叉熵)， 再用Softmax计算百分比概率)
loss = tensorflow.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)

# Adam优化器来最小化误差，学习率0.001
train_op = tensorflow.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

