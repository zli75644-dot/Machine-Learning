第 2 天：机器学习常用工具链
今天的核心目标是：跑通一个标准的机器学习训练流程。
第 1 天我们理解了：
X = 特征
y = 标签
fit = 训练
predict = 预测
第 2 天开始正式使用机器学习常用工具：
numpy
pandas
matplotlib
scikit-learn
今天你不需要深入算法原理，重点是熟悉机器学习项目的基本代码结构。
一、今日学习目标

今天你需要掌握：
使用 numpy 处理数组；
使用 pandas 查看表格数据；
使用 matplotlib 简单画图；
使用 sklearn 加载数据集；
使用 train_test_split 划分训练集和测试集；
使用 fit() 训练模型；
使用 predict() 预测结果；
使用 accuracy_score() 评估模型效果。
二、今天使用的数据集：鸢尾花 Iris 数据集
Iris 是机器学习入门最经典的数据集之一。
它的任务是：根据花的特征，判断花属于哪一种鸢尾花。
1. 特征 X
每朵花有 4 个特征：
sepal length：花萼长度
sepal width：花萼宽度
petal length：花瓣长度
petal width：花瓣宽度

为了方便理解，可以记成：

X = 花的长度和宽度信息
2. 标签 y

鸢尾花一共有 3 个类别：

0：setosa
1：versicolor
2：virginica

也就是：

y = 鸢尾花的类别

这是一个典型的 分类任务。

三、安装依赖

在命令行执行：

pip install numpy pandas matplotlib scikit-learn

如果你使用的是 Anaconda，一般这些库已经自带了。

四、今日完整代码
新建文件：
day02_iris_classification.py

代码如下：

"""
第 2 天：机器学习常用工具链与鸢尾花分类案例。

本案例演示机器学习中的标准流程：
1. 加载数据
2. 查看数据
3. 划分训练集和测试集
4. 创建模型
5. 训练模型
6. 预测结果
7. 评估模型
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class IrisClassificationDemo:
    """
    鸢尾花分类演示类。

    :param test_size: 测试集比例
    :param random_state: 随机种子，保证每次划分数据结果一致
    :param n_neighbors: KNN 模型中的邻居数量
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42, n_neighbors: int = 3):
        """
        初始化鸢尾花分类演示类。

        :param test_size: 测试集比例，例如 0.2 表示 20% 数据作为测试集
        :param random_state: 随机种子，用于保证实验结果可复现
        :param n_neighbors: KNN 算法中的 K 值
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state
        self.n_neighbors = n_neighbors

        self.iris_data = None
        self.feature_names = None
        self.target_names = None

        self.X = None
        self.y = None
        self.df = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.y_pred = None

    def load_data(self):
        """
        加载 sklearn 自带的鸢尾花数据集。

        :return: 无返回值
        """
        self.iris_data = load_iris()

        self.X = self.iris_data.data
        self.y = self.iris_data.target

        self.feature_names = self.iris_data.feature_names
        self.target_names = self.iris_data.target_names

    def build_dataframe(self):
        """
        将 numpy 数组转换为 pandas DataFrame，方便查看数据。

        :return: 鸢尾花数据对应的 DataFrame
        """
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df["label_id"] = self.y
        self.df["label_name"] = self.df["label_id"].apply(lambda index: self.target_names[index])

        return self.df

    def show_basic_info(self):
        """
        打印数据集的基本信息。

        :return: 无返回值
        """
        print("========== 数据集基本信息 ==========")
        print("特征名称：", self.feature_names)
        print("类别名称：", self.target_names)
        print("特征数据形状：", self.X.shape)
        print("标签数据形状：", self.y.shape)

        print("\n========== 前 5 行数据 ==========")
        print(self.df.head())

        print("\n========== 类别数量统计 ==========")
        print(self.df["label_name"].value_counts())

    def split_data(self):
        """
        将数据集划分为训练集和测试集。

        :return: 无返回值
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )

        print("\n========== 数据集划分结果 ==========")
        print("训练集特征形状：", self.X_train.shape)
        print("测试集特征形状：", self.X_test.shape)
        print("训练集标签形状：", self.y_train.shape)
        print("测试集标签形状：", self.y_test.shape)

    def train_model(self):
        """
        训练 KNN 分类模型。

        :return: 无返回值
        """
        self.model.fit(self.X_train, self.y_train)
        print("\n模型训练完成。")

    def predict_test_data(self):
        """
        使用测试集进行预测。

        :return: 测试集预测结果
        """
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def evaluate_model(self):
        """
        评估模型效果。

        :return: 无返回值
        """
        accuracy = accuracy_score(self.y_test, self.y_pred)

        print("\n========== 模型评估结果 ==========")
        print(f"模型准确率：{accuracy:.4f}")

        print("\n========== 分类报告 ==========")
        print(classification_report(
            self.y_test,
            self.y_pred,
            target_names=self.target_names
        ))

        print("\n========== 混淆矩阵 ==========")
        print(confusion_matrix(self.y_test, self.y_pred))

    def predict_single_sample(self, sample: np.ndarray):
        """
        预测单个样本所属的鸢尾花类别。

        :param sample: 单个样本特征，形状应为 [[花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度]]
        :return: 预测类别编号和类别名称
        """
        pred_id = self.model.predict(sample)[0]
        pred_name = self.target_names[pred_id]

        return pred_id, pred_name

    def visualize_data(self):
        """
        使用 matplotlib 简单可视化鸢尾花数据。

        这里选择两个特征：
        1. petal length
        2. petal width

        :return: 无返回值
        """
        plt.figure(figsize=(8, 6))

        plt.scatter(
            self.df["petal length (cm)"],
            self.df["petal width (cm)"],
            c=self.df["label_id"]
        )

        plt.xlabel("petal length (cm)")
        plt.ylabel("petal width (cm)")
        plt.title("Iris Dataset Visualization")
        plt.colorbar(label="class id")

        plt.show()

    def run(self):
        """
        运行完整的机器学习流程。

        :return: 无返回值
        """
        self.load_data()
        self.build_dataframe()
        self.show_basic_info()

        self.split_data()
        self.train_model()
        self.predict_test_data()
        self.evaluate_model()

        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        pred_id, pred_name = self.predict_single_sample(sample)

        print("\n========== 单条样本预测 ==========")
        print("输入样本：", sample)
        print("预测类别编号：", pred_id)
        print("预测类别名称：", pred_name)

        self.visualize_data()


def main():
    """
    主函数。

    :return: 无返回值
    """
    demo = IrisClassificationDemo(
        test_size=0.2,
        random_state=42,
        n_neighbors=3
    )
    demo.run()


if __name__ == "__main__":
    main()
五、代码执行流程解释

整个机器学习流程可以看成下面这条线：

加载数据
  ↓
查看数据
  ↓
划分训练集和测试集
  ↓
创建模型
  ↓
训练模型
  ↓
预测测试集
  ↓
评估模型
  ↓
预测单个新样本
六、重点知识解释
1. numpy 是干什么的？

numpy 主要用来处理数组。

机器学习中的数据通常长这样：

X = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3]
])

你可以把它理解成一个二维表格：

样本	特征1	特征2	特征3	特征4
样本1	5.1 	3.5 	1.4	    0.2
样本2	4.9	    3.0	    1.4	    0.2
样本3	6.2 	3.4	    5.4	    2.3

在机器学习中：

行 = 样本
列 = 特征
2. pandas 是干什么的？

pandas 主要用来处理表格数据。

例如：
 
 
self.df = pd.DataFrame(self.X, columns=self.feature_names)

这句话的作用是：

把 numpy 数组转换成带有列名的表格。

这样更方便查看数据：

print(self.df.head())

head() 表示查看前 5 行。

3. matplotlib 是干什么的？

matplotlib 用来画图。

今天代码中使用：

plt.scatter(
    self.df["petal length (cm)"],
    self.df["petal width (cm)"],
    c=self.df["label_id"]
)

意思是：

横轴：花瓣长度
纵轴：花瓣宽度
颜色：不同类别

你会发现不同类别的鸢尾花，在图中大致分布在不同区域。

这就是机器学习分类的直观基础：

不同类别的数据，特征分布往往有差异。
4. scikit-learn 是干什么的？

scikit-learn 简称 sklearn，是 Python 中最常用的传统机器学习库。

它里面已经封装好了很多模型，例如：

线性回归
逻辑回归
KNN
决策树
随机森林
SVM
KMeans
PCA

今天使用的是：

from sklearn.neighbors import KNeighborsClassifier

也就是 KNN 分类模型。

七、train_test_split 是什么？

这行代码很重要：

self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
    self.X,
    self.y,
    test_size=self.test_size,
    random_state=self.random_state,
    stratify=self.y
)

它的作用是：

把原始数据拆成训练集和测试集。
1. 为什么要拆分数据？

因为模型不能只在它见过的数据上表现好。

我们真正关心的是：

模型遇到新数据时，能不能预测准确。

所以要分成：

训练集：给模型学习用
测试集：检查模型学习效果用
2. test_size 是什么？
test_size=0.2

表示：

20% 数据作为测试集
80% 数据作为训练集

Iris 数据集一共有 150 条数据，所以：

训练集：120 条
测试集：30 条
3. random_state 是什么？
random_state=42

表示随机种子。

因为数据划分是随机的，如果不固定随机种子，每次运行代码，训练集和测试集可能都不一样。

固定后：

每次运行，划分结果一致，方便复现实验。
4. stratify 是什么？
stratify=self.y

表示按标签比例分层抽样。

Iris 有 3 个类别，每个类别 50 条。

加上 stratify=self.y 后，训练集和测试集中各类别比例会尽量保持一致。

这样模型评估更公平。

八、fit、predict、score 的含义
1. fit：训练模型
self.model.fit(self.X_train, self.y_train)

含义：

让模型从训练数据中学习规律。
2. predict：预测结果
self.y_pred = self.model.predict(self.X_test)

含义：

让模型对测试集进行预测。
3. accuracy_score：计算准确率
accuracy = accuracy_score(self.y_test, self.y_pred)

含义：

用真实答案 y_test 和预测答案 y_pred 进行对比，计算预测正确的比例。

例如测试集有 30 条，预测对了 29 条：

accuracy = 29 / 30 = 0.9667
九、KNN 模型简单理解

今天我们先不深入 KNN 数学原理，只需要知道它的核心思想：

看一个新样本离哪些训练样本最近，然后根据最近邻居的类别来判断它属于哪一类。

例如：

一朵新花来了；
模型看它和历史数据中哪些花最像；
如果最近的 3 朵花中，有 2 朵都是 setosa；
那么模型就预测它是 setosa。

所以：

KNeighborsClassifier(n_neighbors=3)

表示：

看最近的 3 个邻居。
十、今天的核心代码片段
1. 加载数据
from sklearn.datasets import load_iris

iris_data = load_iris()
X = iris_data.data
y = iris_data.target
2. 查看数据形状
print(X.shape)
print(y.shape)

输出大概是：

(150, 4)
(150,)

含义是：

X 有 150 行，4 列
y 有 150 个标签
3. 划分数据集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
4. 创建模型
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
5. 训练模型
model.fit(X_train, y_train)
6. 预测结果
y_pred = model.predict(X_test)
7. 计算准确率
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
十一、运行后你应该重点看什么？

运行代码后，你重点看这几个地方。

1. 数据形状
特征数据形状：(150, 4)
标签数据形状：(150,)

说明：

有 150 条样本，每条样本有 4 个特征。
2. 训练集和测试集形状
训练集特征形状：(120, 4)
测试集特征形状：(30, 4)

说明：

120 条数据用于训练，30 条数据用于测试。
3. 模型准确率

可能输出类似：

模型准确率：0.9667

说明：

测试集 30 条数据中，大约预测对了 29 条。
4. 单条样本预测
输入样本：[[5.1 3.5 1.4 0.2]]
预测类别编号：0
预测类别名称：setosa

说明模型认为这朵花是：

setosa
十二、今天你要真正掌握的机器学习标准模板

以后大部分传统机器学习代码，都离不开这个模板：

"""
机器学习标准训练流程。
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. 准备特征和标签
X = ...
y = ...

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 3. 创建模型
model = ...

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 预测结果
y_pred = model.predict(X_test)

# 6. 评估模型
accuracy = accuracy_score(y_test, y_pred)
print("准确率：", accuracy)

这就是机器学习最基本的骨架。

十三、今日练习
练习 1：理解数据结构

请回答：

Iris 数据集中，X 有多少行？多少列？
每一行代表什么？
每一列代表什么？
y 代表什么？
练习 2：修改测试集比例

把代码中的：

test_size=0.2

改成：

test_size=0.3

观察：

训练集数量变成多少？
测试集数量变成多少？
模型准确率有没有变化？
练习 3：修改 KNN 的 K 值

把：

n_neighbors=3

分别改成：

n_neighbors=1
n_neighbors=5
n_neighbors=7

观察模型准确率是否变化。

练习 4：预测新的鸢尾花

修改这行：

sample = np.array([[5.1, 3.5, 1.4, 0.2]])

改成：

sample = np.array([[6.0, 3.0, 4.8, 1.8]])

观察预测结果是什么。

练习 5：只使用两个特征训练模型

现在的特征是 4 个：

花萼长度
花萼宽度
花瓣长度
花瓣宽度

你可以尝试只使用后两个特征：

self.X = self.iris_data.data[:, 2:4]

也就是只使用：

花瓣长度
花瓣宽度

再训练模型，看看准确率是否变化。

十四、今日总结

今天你完成了机器学习中非常重要的一步：跑通标准训练流程。

你现在应该能看懂下面这段流程：

load_iris()
    ↓
X, y
    ↓
train_test_split()
    ↓
X_train, X_test, y_train, y_test
    ↓
KNeighborsClassifier()
    ↓
model.fit()
    ↓
model.predict()
    ↓
accuracy_score()

今天最重要的一句话是：

机器学习代码的基本流程不是先写复杂算法，而是先把数据、模型、训练、预测、评估这几个环节串起来。
十五、第 2 天学习检查表

你可以按下面标准检查自己是否学会：

检查项	是否掌握
能说出 numpy 的作用	
能说出 pandas 的作用	
能说出 matplotlib 的作用	
能说出 sklearn 的作用	
能理解 X 和 y	
能理解训练集和测试集	
能理解 fit 的作用	
能理解 predict 的作用	
能理解 accuracy_score 的作用	
能独立运行 Iris 分类代码