第 4 天：梯度下降与模型训练原理

今天开始理解机器学习里真正的“学习”过程。

前 3 天你已经知道：

X = 特征
y = 标签
model = 模型
fit = 训练
predict = 预测
w = 权重
b = 偏置

今天要重点解决这个问题：

模型一开始并不知道 w 和 b 应该是多少，那它是怎么通过训练找到合适参数的？

答案就是：

损失函数 + 梯度下降
一、今日学习目标

今天你需要掌握 7 个核心概念：

什么是损失函数；
什么是预测误差；
什么是梯度；
什么是梯度下降；
什么是学习率 learning_rate；
什么是迭代次数 epoch；
不使用 sklearn，自己用 numpy 实现一个简单线性回归训练过程。
二、先回顾第 3 天的线性回归

第 3 天我们学过线性回归：

y = wx + b

如果是房价预测：

预测房价 = 面积 × w + b

比如：

面积 x = 100
权重 w = 3
偏置 b = -20

那么预测值就是：

预测房价 = 100 × 3 - 20 = 280

问题是：

w = 3，b = -20 是怎么来的？

不是人手动写死的，而是模型通过训练数据慢慢学出来的。

三、机器学习的训练过程

模型训练可以理解成下面这个过程：

第 1 步：先随机给 w 和 b 一个初始值
第 2 步：用当前 w 和 b 进行预测
第 3 步：计算预测值和真实值之间的误差
第 4 步：根据误差调整 w 和 b
第 5 步：重复很多次，让误差越来越小

也就是：

乱猜参数
    ↓
计算错误
    ↓
调整参数
    ↓
再计算错误
    ↓
继续调整参数
    ↓
直到模型预测越来越准

这就是机器学习中的“学习”。

四、什么是损失函数？

损失函数就是用来衡量模型预测得有多差。

比如真实房价是：

真实值 y_true = 300

模型预测：

预测值 y_pred = 280

那么误差是：

误差 = 真实值 - 预测值 = 300 - 280 = 20

但是一个数据集里有很多样本，我们不能只看一条样本的误差，所以需要一个总指标衡量模型整体预测效果。

线性回归常用的损失函数是 MSE，均方误差：

你可以先不用纠结公式细节，只需要理解：

J(w, b) 表示当前 w 和 b 对应的整体错误程度。
J(w, b) 越大，说明模型越差。
J(w, b) 越小，说明模型越好。

机器学习训练的目标就是：

找到一组 w 和 b，让损失函数 J(w, b) 尽可能小。
五、什么是梯度？

梯度可以先简单理解为：

告诉模型参数应该往哪个方向调整，损失函数会下降得更快。

比如当前模型预测房价总是偏低，那么模型可能需要把 w 调大一点。

如果当前模型预测房价总是偏高，那么模型可能需要把 w 调小一点。

梯度的作用就是告诉模型：

w 应该变大还是变小？
b 应该变大还是变小？
每次应该调整多少？
六、什么是梯度下降？

梯度下降就是：

沿着损失函数下降最快的方向，一步一步调整参数，让损失越来越小。

参数更新公式可以写成：

这里：

w：权重
b：偏置
α：学习率，也就是 learning_rate
∂J/∂w：损失函数对 w 的梯度
∂J/∂b：损失函数对 b 的梯度

通俗理解：

新参数 = 旧参数 - 学习率 × 梯度
七、什么是学习率 learning_rate？

学习率控制每次参数更新的步子大小。

例如：

learning_rate = 0.01

表示模型每次按照梯度方向调整一点点。

1. 学习率太大

如果学习率太大，模型可能一步迈得太远，直接跨过最优点。

表现为：

损失函数忽大忽小，甚至越来越大。
2. 学习率太小

如果学习率太小，模型每次调整得太慢。

表现为：

训练很稳定，但是收敛特别慢。
3. 合适的学习率

合适的学习率应该让损失函数：

整体逐渐下降，并最终趋于稳定。

你可以先记住：

学习率不是越大越好，也不是越小越好，而是要合适。
八、什么是 epoch？

epoch 表示训练轮数。

例如：

epochs = 1000

表示模型会完整训练 1000 轮。

每一轮都做：

预测
计算损失
计算梯度
更新 w 和 b

所以：

epoch 越大，模型训练次数越多；
但 epoch 过大，也可能浪费时间，甚至过拟合。
九、手动推导一元线性回归训练逻辑

我们先用最简单的一元线性回归：

y = wx + b

有一组训练数据：

面积 x	房价 y
60	150
70	180
80	210
90	240
100	270
110	300
120	330

模型要学习：

房价 = 面积 × w + b

训练前：

w = 0
b = 0

第一次预测：

面积 60，预测房价 = 60 × 0 + 0 = 0
面积 70，预测房价 = 70 × 0 + 0 = 0
面积 80，预测房价 = 80 × 0 + 0 = 0

这明显很差。

然后模型计算误差，发现预测值整体偏低，于是调整参数。

经过很多轮训练后，模型可能学到：

w ≈ 3
b ≈ -30

那么：

面积 100，预测房价 = 100 × 3 - 30 = 270

就比较接近真实规律。

十、今日完整代码：不用 sklearn，手写梯度下降

新建文件：

day04_gradient_descent.py

代码如下：

"""
第 4 天：梯度下降与模型训练原理。

本案例不使用 sklearn 的 LinearRegression，
而是使用 numpy 手动实现一个简单的一元线性回归模型。

核心目标：
1. 理解 w 和 b 是如何被训练出来的
2. 理解损失函数 MSE
3. 理解梯度下降
4. 理解 learning_rate 和 epochs
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionByGradientDescent:
    """
    使用梯度下降实现一元线性回归。

    :param learning_rate: 学习率，控制每次参数更新的步长
    :param epochs: 训练轮数
    :return: 无返回值
    """

    def __init__(self, learning_rate: float = 0.0001, epochs: int = 1000):
        """
        初始化线性回归模型。

        :param learning_rate: 学习率，例如 0.0001
        :param epochs: 训练轮数，例如 1000
        :return: 无返回值
        """
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w = 0.0
        self.b = 0.0

        self.loss_history = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        根据当前 w 和 b 进行预测。

        模型公式：
        y_pred = w * X + b

        :param X: 输入特征数组
        :return: 预测结果数组
        """
        return self.w * X + self.b

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算均方误差 MSE。

        :param y_true: 真实标签
        :param y_pred: 模型预测值
        :return: 当前模型的 MSE 损失
        """
        loss = np.mean((y_true - y_pred) ** 2)
        return loss

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练模型。

        训练过程：
        1. 根据当前 w 和 b 计算预测值
        2. 计算损失函数
        3. 计算 w 和 b 的梯度
        4. 使用梯度下降更新 w 和 b

        :param X: 训练特征
        :param y: 训练标签
        :return: 无返回值
        """
        n = len(X)

        for epoch in range(self.epochs):
            y_pred = self.predict(X)

            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            dw = (-2 / n) * np.sum(X * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            if epoch % 100 == 0:
                print(
                    f"第 {epoch} 轮训练，"
                    f"loss={loss:.4f}，"
                    f"w={self.w:.4f}，"
                    f"b={self.b:.4f}"
                )

    def show_parameters(self):
        """
        打印模型训练后的参数。

        :return: 无返回值
        """
        print("\n========== 模型最终参数 ==========")
        print(f"权重 w：{self.w:.4f}")
        print(f"偏置 b：{self.b:.4f}")

    def plot_loss_curve(self):
        """
        绘制损失函数下降曲线。

        :return: 无返回值
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training Loss Curve")
        plt.show()

    def plot_fit_result(self, X: np.ndarray, y: np.ndarray):
        """
        绘制真实数据点和模型拟合直线。

        :param X: 输入特征
        :param y: 真实标签
        :return: 无返回值
        """
        y_pred = self.predict(X)

        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, label="True Data")
        plt.plot(X, y_pred, label="Fitted Line")
        plt.xlabel("Area")
        plt.ylabel("Price")
        plt.title("Linear Regression Fit Result")
        plt.legend()
        plt.show()


class HousePriceTrainingDemo:
    """
    房价训练演示类。

    :param None: 初始化时不需要外部参数
    :return: 无返回值
    """

    def __init__(self):
        """
        初始化房价训练演示类。

        :return: 无返回值
        """
        self.X = None
        self.y = None
        self.model = LinearRegressionByGradientDescent(
            learning_rate=0.0001,
            epochs=1000
        )

    def build_dataset(self):
        """
        构造一元线性回归训练数据。

        X 表示房子面积。
        y 表示房价，单位万元。

        :return: 无返回值
        """
        self.X = np.array([60, 70, 80, 90, 100, 110, 120], dtype=float)
        self.y = np.array([150, 180, 210, 240, 270, 300, 330], dtype=float)

    def train(self):
        """
        训练模型。

        :return: 无返回值
        """
        self.model.fit(self.X, self.y)

    def predict_new_house(self, area: float):
        """
        预测新房子的价格。

        :param area: 房子面积，单位平方米
        :return: 预测房价
        """
        new_x = np.array([area], dtype=float)
        predicted_price = self.model.predict(new_x)[0]
        return predicted_price

    def run(self):
        """
        运行完整训练流程。

        :return: 无返回值
        """
        print("========== 第 4 天：梯度下降训练线性回归 ==========")

        self.build_dataset()

        print("\n========== 训练数据 ==========")
        for x_value, y_value in zip(self.X, self.y):
            print(f"面积：{x_value:.0f} 平方米，房价：{y_value:.0f} 万元")

        self.train()

        self.model.show_parameters()

        predicted_price = self.predict_new_house(area=95)

        print("\n========== 新房子预测 ==========")
        print("输入面积：95 平方米")
        print(f"预测房价：{predicted_price:.2f} 万元")

        self.model.plot_loss_curve()
        self.model.plot_fit_result(self.X, self.y)


def main():
    """
    主函数。

    :return: 无返回值
    """
    demo = HousePriceTrainingDemo()
    demo.run()


if __name__ == "__main__":
    main()
十一、代码重点解释
1. 初始化 w 和 b
self.w = 0.0
self.b = 0.0

这表示模型一开始什么都不知道。

初始公式是：

y_pred = 0 * x + 0

所以无论输入什么，预测值都是 0。

2. 预测函数
def predict(self, X: np.ndarray) -> np.ndarray:
    return self.w * X + self.b

这就是线性回归公式：

预测值 = w × X + b
3. 损失函数
loss = np.mean((y_true - y_pred) ** 2)

意思是：

计算所有样本的预测误差；
把误差平方；
再求平均值。

也就是 MSE。

4. 计算梯度
dw = (-2 / n) * np.sum(X * (y - y_pred))
db = (-2 / n) * np.sum(y - y_pred)

这两行代码的作用是：

计算当前 w 和 b 应该往哪个方向调整。

你现在不需要完全手推公式，但需要知道：

dw 表示 w 的调整方向；
db 表示 b 的调整方向。
5. 更新参数
self.w = self.w - self.learning_rate * dw
self.b = self.b - self.learning_rate * db

这就是梯度下降最核心的代码。

含义是：

用当前参数减去 学习率 × 梯度。

如果这两行看懂了，你就已经理解了机器学习训练的核心。

6. 保存损失变化
self.loss_history.append(loss)

每一轮训练都会保存当前损失。

后面可以画出损失下降曲线：

plt.plot(self.loss_history)

如果训练正常，你应该看到损失整体下降。

十二、运行后重点观察什么？

运行代码后，重点看类似输出：

第 0 轮训练，loss=60300.0000，w=4.5000，b=0.0480
第 100 轮训练，loss=...
第 200 轮训练，loss=...
...

你要观察 3 个变化：

loss 是否越来越小；
w 是否逐渐趋于稳定；
b 是否逐渐趋于稳定。

如果 loss 越来越小，说明模型正在学习。

如果 loss 越来越大，通常说明：

learning_rate 太大了。
十三、学习率实验

你可以修改：

learning_rate=0.0001

分别改成：

learning_rate=0.001
learning_rate=0.00001
learning_rate=0.01

观察现象。

1. learning_rate = 0.00001

可能表现为：

loss 下降很慢。

说明步子太小。

2. learning_rate = 0.0001

通常比较稳定。

说明步子比较合适。

3. learning_rate = 0.01

可能表现为：

loss 变得非常大，甚至出现 nan。

说明步子太大，模型直接跑偏了。

十四、梯度下降的生活类比

你可以把梯度下降理解成“下山”。

你站在一座山上；
目标是走到山谷最低点；
你不知道最低点在哪里；
但是你能感受到当前哪个方向更低；
于是你每次往更低的方向走一步；
走很多步后，就越来越接近山谷底部。

对应到机器学习：

山的高度 = 损失函数
当前位置 = 当前参数 w 和 b
下山方向 = 梯度方向
每一步大小 = 学习率
走多少步 = epoch
山谷底部 = 最优参数
十五、今天容易混淆的几个点
1. 误差和损失不是完全一样

误差通常指一条样本：

真实值 - 预测值

损失通常指整个数据集的总体错误程度，例如 MSE。

2. 梯度不是误差

误差是模型错了多少。

梯度是参数应该怎么调整。

3. 学习率不是训练次数

学习率控制每一步走多大。

epoch 控制总共走多少步。

4. 损失越小不一定永远越好

在训练集上损失越小通常表示拟合更好。

但如果模型太复杂，训练集损失很小，测试集效果却很差，就可能是过拟合。

这个问题后面第 14 天会详细讲。

十六、今天必须掌握的核心代码

下面这几行就是今天的灵魂代码：

y_pred = self.predict(X)

loss = np.mean((y - y_pred) ** 2)

dw = (-2 / n) * np.sum(X * (y - y_pred))
db = (-2 / n) * np.sum(y - y_pred)

self.w = self.w - self.learning_rate * dw
self.b = self.b - self.learning_rate * db

对应的含义是：

先预测；
再算损失；
再算梯度；
最后更新参数。
十七、今日练习
练习 1：解释训练过程

请用自己的话解释下面流程：

初始化 w 和 b
    ↓
计算预测值
    ↓
计算损失
    ↓
计算梯度
    ↓
更新 w 和 b
练习 2：修改学习率

把代码中的：

learning_rate=0.0001

分别改成：

learning_rate=0.00001
learning_rate=0.001
learning_rate=0.01

观察：

loss 是否下降？
loss 下降速度是否变化？
是否出现 loss 变大或者 nan？
练习 3：修改训练轮数

把：

epochs=1000

分别改成：

epochs=100
epochs=500
epochs=3000

观察最终的：

w
b
loss
预测房价

是否发生变化。

练习 4：增加预测样本

使用模型预测下面房子的价格：

面积 = 75
面积 = 95
面积 = 135

你可以这样调用：

print(demo.predict_new_house(75))
print(demo.predict_new_house(95))
print(demo.predict_new_house(135))
练习 5：增加训练数据

把训练数据改成：

self.X = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140], dtype=float)
self.y = np.array([120, 150, 180, 210, 240, 270, 300, 330, 360, 390], dtype=float)

重新训练，观察模型最终学到的 w 和 b。

十八、今日学习检查表
检查项	是否掌握
能理解损失函数的作用	
能理解 MSE 是什么	
能理解梯度的作用	
能理解梯度下降的基本思想	
能理解 learning_rate 是步长	
能理解 epoch 是训练轮数	
能看懂 w = w - learning_rate * dw	
能观察 loss 是否下降	
能手写一个简单线性回归训练过程	
十九、今日总结

第 4 天最重要的是这句话：

机器学习中的训练，本质上就是通过损失函数衡量模型错误，再通过梯度下降不断调整参数，让损失函数越来越小。

你现在应该能理解：

model.fit(X, y)

背后大致发生了什么：

初始化参数
    ↓
计算预测值
    ↓
计算损失
    ↓
计算梯度
    ↓
更新参数
    ↓
重复很多轮
    ↓
得到最终模型

本章自己思考的问题，
w 和 b的变化
            dw = (-2/n) * np.sum((y - y_pred) * X)
            db = (-2/n) * np.sum(y - y_pred)
这个是偏导，其实就是(y - y_pred)^2 的偏导，
参数是下面这样变化的
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
        所以它的权重可以会变self.learning_rate * dw的值
我们的loss是
Loss = 1/n * Σ(y - y_pred)^2
他和w之间是存在联系的，当loss大，意味着预测值和真实值之间的差距大，
而他不会影响到w，但是他能看出来loss的变化，正常情况下希望loss越来越小，那么w是加learning_rate * dw还是减learning_rate * dw
取决于预测值和真实值的相差，
那么loss起到的作用是当看到 Loss 在下降，就知道模型整体在变好。