import matplotlib.pyplot as plt
import numpy


# 创建数据
x = numpy.linspace(-2, 2, 100)   # 起点-2， 终点2，等分为100份的一条直线
y1 = 3 * x + 4
y2 = x ** 2

# 创建图像
# plt.plot(x, y1)
# plt.show()

# 创建多张定制图像
# 创建第一张图
plt.figure(num=1, figsize=(7, 6))
plt.plot(x, y1)
plt.plot(x, y2, color="red", linewidth=3.0, linestyle="--")

# 创建第二章图
plt.figure(num=2)
plt.plot(x, y2, color="green")

# 图像展示
plt.show()


