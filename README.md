本项目为记录学习TensorFlow

环境介绍：  TensorFlow：1.13.1     Python：3.6     OS: WINDOW7 64位

01-tensorboard.py
线性回归方程y = W * x + b利用TensorFlow求解，并将日志保存下log文件下，在当前目录下打开cmd窗口，
输入tensorboard --logdir=log可以在http://localhost:6006中看到TensorFlow的工作细节流程

02-matplotlib.py
利用Python数据库matplotlib跟numpy绘制出y1 = 3 * x + 4跟y2 = x ** 2的函数图像，并且可以定制图像的内容

03-LR_using_GR.py
用梯度下降的优化方法来快速解决线性回归问题,给定100个(x,y)的坐标点，利用matplotlib绘制出图像，用TensorFlow拟合
线性回归方程

04-activation_func.py
使用TensorFlow的激活函数结合numpy绘制出非线性的函数图像，绘制了y = 1 / float(1 + numpy.exp(-x))， y = x * (x > 0)，
y = (numpy.exp(x) - numpy.exp(-x)) / float(numpy.exp(x) + numpy.exp(-x))，y = numpy.log(1 + numpy.exp(x))
的图像，经过TensorFlow的激活函数处理的各个Y值
