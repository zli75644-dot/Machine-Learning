第 3 天：线性回归 Linear Regression
今天开始进入真正的机器学习算法。第 1 天你理解了：

X = 特征
y = 标签
model = 模型
fit = 训练
predict = 预测

第 2 天你跑通了 sklearn 的标准训练流程。

第 3 天要解决的问题是：

模型到底在学习什么？
为什么输入几个特征，就能预测一个连续数值？

今天重点学习 线性回归，它主要用于预测连续数值，比如房价、销量、温度、收入、成本等。

一、今日学习目标

今天你需要掌握 7 个内容：

理解什么是回归任务；
理解线性回归的基本思想；
理解权重 w 和偏置 b；
理解预测值 y_pred 和真实值 y_true；
理解误差和损失函数 MSE；
使用 sklearn 训练线性回归模型；
能够查看模型学到的权重和偏置。
二、什么是回归任务？

机器学习任务大致分为两类：

分类：预测类别
回归：预测连续数值
1. 分类任务

分类任务输出的是类别。

例如：

判断邮件是否垃圾邮件：是 / 否
判断图片是猫还是狗：猫 / 狗
判断评论是好评还是差评：好评 / 差评

输出通常是：

0、1
猫、狗
好评、差评
2. 回归任务

回归任务输出的是连续数值。

例如：

预测房价：280 万
预测销量：1360 件
预测温度：31.5 ℃
预测用户消费金额：258.8 元

线性回归就是最经典的回归算法。

三、线性回归的核心思想

线性回归认为：

一个结果 y，可以由多个特征 x 按照一定权重加起来得到。

最简单的一元线性回归是：

这里：

x：输入特征
w：权重
b：偏置
y：预测结果

例如预测房价：

房价 = 面积 × 权重 + 偏置

如果考虑多个特征，就是多元线性回归：

房价 = 面积 × w1 + 房间数 × w2 + 楼层 × w3 + b

写成机器学习形式：

y = w1*x1 + w2*x2 + w3*x3 + b
四、用房价预测理解线性回归

假设有一批房子数据：

面积	房间数	楼层	房价
60  	1	    6   	150
70  	2	    8	    180
80	    2       10      210
90	    2	    12  	240
100	    3	    15	    280
120	    3   	18  	340
150	    4   	25	    460

这里：

X = 面积、房间数、楼层
y = 房价

模型要学习一个公式：

预测房价 = 面积 × w1 + 房间数 × w2 + 楼层 × w3 + b

比如模型最后学到了：

w1 = 2.8
w2 = 12.5
w3 = 1.6
b = -40

那么新房子：

面积 = 110
房间数 = 3
楼层 = 16

预测过程大概是：

预测房价 = 110 × 2.8 + 3 × 12.5 + 16 × 1.6 - 40

也就是说，线性回归模型本质上是在学习：

每个特征对最终结果的影响程度。
五、w 和 b 怎么理解？
1. 权重 w

w 表示某个特征的重要程度，也可以理解为这个特征对预测结果的影响。

例如：

面积对应的权重越大，说明面积对房价影响越明显；
房间数对应的权重越大，说明房间数对房价影响越明显；
楼层对应的权重越小，说明楼层对房价影响较弱。

在 sklearn 里，权重保存在：

model.coef_
2. 偏置 b

b 是基础偏移量。

即使所有特征为 0，模型也会有一个基础输出。

在 sklearn 里，偏置保存在：

model.intercept_
六、误差是什么？

模型刚开始不可能预测得完全准确。

例如真实房价是：

真实房价 y_true = 300 万

模型预测：

预测房价 y_pred = 280 万

那么误差就是：

误差 = 真实值 - 预测值 = 300 - 280 = 20

机器学习训练的目标就是：

让预测值尽可能接近真实值。
七、MSE 均方误差

线性回归常用的损失函数是 MSE。

MSE 的全称是：

Mean Squared Error

中文叫：

均方误差

它的计算逻辑是：

1. 每条样本计算：真实值 - 预测值
2. 将误差平方
3. 对所有样本的平方误差求平均

公式可以理解为：

MSE = 所有样本的误差平方之和 / 样本数量

为什么要平方？

因为有的误差是正数，有的误差是负数，如果直接相加，可能互相抵消。

例如：

第 1 条误差：+20
第 2 条误差：-20

直接相加是 0，但模型明明预测错了。

平方后：

20² = 400
(-20)² = 400

错误就不会被抵消。

八、今日完整代码

新建文件：

day03_linear_regression.py

代码如下：

"""
第 3 天：线性回归 Linear Regression。

本案例使用一个小型房价数据集，演示线性回归的完整流程：
1. 构造数据
2. 划分训练集和测试集
3. 训练线性回归模型
4. 查看模型权重和偏置
5. 预测测试集
6. 计算回归评估指标
7. 预测单套新房价格
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class HousePriceLinearRegression:
    """
    房价线性回归预测类。

    :param test_size: 测试集比例
    :param random_state: 随机种子，保证每次划分数据结果一致
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化房价线性回归预测类。

        :param test_size: 测试集比例，例如 0.2 表示 20% 数据作为测试集
        :param random_state: 随机种子，保证每次运行结果一致
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state

        self.df = None

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.model = LinearRegression()
        self.y_pred = None

    def build_dataset(self):
        """
        构造房价数据集。

        数据字段说明：
        area: 面积，单位平方米
        room_count: 房间数量
        floor: 楼层
        price: 房价，单位万元

        :return: 房价数据 DataFrame
        """
        data = {
            "area": [60, 70, 80, 90, 100, 110, 120, 130, 150, 160],
            "room_count": [1, 2, 2, 2, 3, 3, 3, 4, 4, 5],
            "floor": [6, 8, 10, 12, 15, 16, 18, 20, 25, 28],
            "price": [150, 180, 210, 240, 280, 310, 340, 380, 460, 500]
        }

        self.df = pd.DataFrame(data)
        return self.df

    def prepare_features_and_label(self):
        """
        准备特征 X 和标签 y。

        X 表示模型输入特征：
        1. area
        2. room_count
        3. floor

        y 表示模型需要预测的目标：
        1. price

        :return: 无返回值
        """
        feature_columns = ["area", "room_count", "floor"]
        label_column = "price"

        self.X = self.df[feature_columns]
        self.y = self.df[label_column]

    def split_data(self):
        """
        划分训练集和测试集。

        :return: 无返回值
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def train_model(self):
        """
        训练线性回归模型。

        :return: 无返回值
        """
        self.model.fit(self.X_train, self.y_train)

    def show_model_parameters(self):
        """
        打印模型学习到的权重和偏置。

        coef_ 表示每个特征对应的权重。
        intercept_ 表示模型的偏置项。

        :return: 无返回值
        """
        print("\n========== 模型参数 ==========")

        for feature_name, weight in zip(self.X.columns, self.model.coef_):
            print(f"{feature_name} 对应的权重 w：{weight:.4f}")

        print(f"模型偏置 b：{self.model.intercept_:.4f}")

    def predict_test_data(self):
        """
        使用测试集进行预测。

        :return: 测试集预测结果
        """
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def evaluate_model(self):
        """
        评估线性回归模型效果。

        常见回归评估指标：
        1. MSE: 均方误差，越小越好
        2. RMSE: 均方根误差，越小越好
        3. MAE: 平均绝对误差，越小越好
        4. R2: 决定系数，越接近 1 越好

        :return: 无返回值
        """
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)

        print("\n========== 模型评估结果 ==========")
        print(f"MSE 均方误差：{mse:.4f}")
        print(f"RMSE 均方根误差：{rmse:.4f}")
        print(f"MAE 平均绝对误差：{mae:.4f}")
        print(f"R2 决定系数：{r2:.4f}")

    def show_prediction_detail(self):
        """
        打印测试集真实值和预测值对比。

        :return: 无返回值
        """
        result_df = self.X_test.copy()
        result_df["真实房价"] = self.y_test.values
        result_df["预测房价"] = self.y_pred
        result_df["误差"] = result_df["真实房价"] - result_df["预测房价"]

        print("\n========== 测试集预测详情 ==========")
        print(result_df)

    def predict_single_house(self, area: float, room_count: int, floor: int):
        """
        预测单套房子的价格。

        :param area: 房子面积，单位平方米
        :param room_count: 房间数量
        :param floor: 房子所在楼层
        :return: 预测房价
        """
        new_house = pd.DataFrame(
            [[area, room_count, floor]],
            columns=["area", "room_count", "floor"]
        )

        predicted_price = self.model.predict(new_house)[0]
        return predicted_price

    def run(self):
        """
        运行完整的线性回归流程。

        :return: 无返回值
        """
        print("========== 第 3 天：线性回归房价预测 ==========")

        self.build_dataset()

        print("\n========== 原始数据 ==========")
        print(self.df)

        self.prepare_features_and_label()
        self.split_data()
        self.train_model()

        self.show_model_parameters()

        self.predict_test_data()
        self.evaluate_model()
        self.show_prediction_detail()

        predicted_price = self.predict_single_house(
            area=95,
            room_count=2,
            floor=13
        )

        print("\n========== 单套房子预测 ==========")
        print("输入特征：面积=95，房间数=2，楼层=13")
        print(f"预测房价：{predicted_price:.2f} 万元")


def main():
    """
    主函数。

    :return: 无返回值
    """
    demo = HousePriceLinearRegression(
        test_size=0.2,
        random_state=42
    )

    demo.run()


if __name__ == "__main__":
    main()
九、代码重点解释
1. 构造数据集
data = {
    "area": [60, 70, 80, 90, 100],
    "room_count": [1, 2, 2, 2, 3],
    "floor": [6, 8, 10, 12, 15],
    "price": [150, 180, 210, 240, 280]
}

这里：

area、room_count、floor 是特征 X
price 是标签 y
2. 准备 X 和 y
feature_columns = ["area", "room_count", "floor"]
label_column = "price"

self.X = self.df[feature_columns]
self.y = self.df[label_column]

等价于告诉模型：

请你根据面积、房间数、楼层，去学习房价。
3. 创建线性回归模型
self.model = LinearRegression()

这就是 sklearn 封装好的线性回归模型。

4. 训练模型
self.model.fit(self.X_train, self.y_train)

模型会尝试学习一个公式：

price = area*w1 + room_count*w2 + floor*w3 + b
5. 查看权重和偏置
self.model.coef_
self.model.intercept_

其中：

coef_：模型学到的各个特征权重
intercept_：模型学到的偏置

例如输出可能类似：

area 对应的权重 w：2.95
room_count 对应的权重 w：8.43
floor 对应的权重 w：1.22
模型偏置 b：-38.50

可以理解为：

面积每增加 1 平方米，房价大约增加 2.95 万；
房间数每增加 1 个，房价大约增加 8.43 万；
楼层每增加 1 层，房价大约增加 1.22 万。

注意：这是基于当前小数据集学出来的结果，不代表真实房地产规律。

十、回归任务常用评估指标

分类任务常用：

accuracy
precision
recall
f1-score

回归任务常用：

MSE
RMSE
MAE
R2
1. MSE：均方误差
MSE = 平均平方误差

特点：

越小越好
对大误差更敏感

例如预测错得特别离谱，MSE 会明显变大。

2. RMSE：均方根误差
RMSE = MSE 开平方

特点：

越小越好
单位和原始标签一致

如果房价单位是万元，那么 RMSE 的单位也是万元。

3. MAE：平均绝对误差
MAE = 平均绝对误差

它直接看平均预测偏差。

例如：

MAE = 15

可以理解为：

模型平均预测误差大约是 15 万元。
4. R2：决定系数
R2 越接近 1，说明模型拟合效果越好。

大致可以这样理解：

R2 = 1：预测非常好
R2 = 0：模型还不如直接猜平均值
R2 < 0：模型效果很差
十一、今天必须理解的核心关系

线性回归训练前：

模型不知道 w 和 b 应该是多少。

训练时：

模型根据 X 和 y，不断寻找更合适的 w 和 b。

训练后：

模型得到一个可以预测的公式。

最终变成：

输入：面积、房间数、楼层
输出：预测房价
十二、你可以这样理解线性回归

线性回归就像是在找一个“定价公式”。

例如：

房价 = 面积贡献 + 房间数贡献 + 楼层贡献 + 基础价格

其中：

面积贡献 = 面积 × 面积权重
房间数贡献 = 房间数 × 房间数权重
楼层贡献 = 楼层 × 楼层权重
基础价格 = 偏置 b

所以，线性回归不是简单记住每套房子的价格，而是学出一个通用公式，用来预测没见过的新房子。

十三、今日练习
练习 1：解释 X 和 y

请回答：

在今天的房价预测代码中：
哪些字段是 X？
哪个字段是 y？
练习 2：解释 w 和 b

请用自己的话解释：

w 是什么？
b 是什么？
为什么线性回归要学习 w 和 b？
练习 3：新增一个特征

给数据集新增一个字段：

distance_to_center：距离市中心距离

例如：

"distance_to_center": [12, 10, 9, 8, 6, 5, 4, 3, 2, 1]

然后把特征列改成：

feature_columns = ["area", "room_count", "floor", "distance_to_center"]

重新训练模型，观察权重变化。

练习 4：修改预测样本

使用模型预测下面几套房子：

面积=85，房间数=2，楼层=9
面积=115，房间数=3，楼层=17
面积=140，房间数=4，楼层=22

你可以调用：

predict_single_house(area=85, room_count=2, floor=9)
练习 5：观察误差

运行代码后，看这一列：

误差 = 真实房价 - 预测房价

请思考：

误差为正数，说明模型预测高了还是低了？
误差为负数，说明模型预测高了还是低了？

答案：

误差为正数：真实值 > 预测值，说明模型预测低了。
误差为负数：真实值 < 预测值，说明模型预测高了。
十四、今日加深理解：为什么线性回归适合入门？

因为线性回归非常直观。

它的核心就是：

给每个特征分配一个权重，然后加起来得到预测值。

例如：

房价 = 面积 × 权重 + 房间数 × 权重 + 楼层 × 权重 + 偏置

你可以把线性回归看成机器学习里的“基础模型”。

后面很多复杂模型，本质上也在做类似的事：

输入特征 → 学习参数 → 输出预测

只不过复杂模型学习的关系不是简单直线，而是更加复杂的非线性关系。

十五、第 3 天学习检查表
检查项	是否掌握
能区分分类和回归	
能说出线性回归解决什么问题	
能理解 y = wx + b	
能理解多个特征对应多个权重	
能理解 coef_ 是权重	
能理解 intercept_ 是偏置	
能理解预测值和真实值的误差	
能理解 MSE 越小越好	
能独立运行线性回归代码	
能预测一条新样本	
十六、今日总结

第 3 天最重要的是理解这句话：

线性回归就是让模型从数据中学习每个特征的权重 w 和整体偏置 b，然后用这些参数组合出一个预测公式。

你现在应该能看懂：


model = LinearRegression()
model.fit(X_train, y_train)

print(model.coef_)
print(model.intercept_)

y_pred = model.predict(X_test)

这几行代码背后的含义：

创建线性回归模型；
用训练数据学习权重和偏置；
查看模型学到的规律；
用模型预测新数据。

代码中存在问题的地方：
