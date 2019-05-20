import tensorflow

W = tensorflow.Variable(2.0, dtype=tensorflow.float32, name="Weight")   # 权重
b = tensorflow.Variable(1.0, dtype=tensorflow.float32, name="Bias")     # 偏差
x = tensorflow.placeholder(dtype=tensorflow.float32, name="Input")      # 输入

with tensorflow.name_scope("Output"):
    y = W * x + b

# 定义保存日志的路径
path = ".\log"

# 创建用于初始化所有变量Variable的操作
init = tensorflow.global_variables_initializer()
path = r"G:\python_project\tensorflow\logs"
# 创建session
with tensorflow.Session() as sess:
    sess.run(init)   # 初始化变量
    writer = tensorflow.summary.FileWriter(path, sess.graph)
    result = sess.run(y, {x: 3.0})
    print("y = %s" % result)

