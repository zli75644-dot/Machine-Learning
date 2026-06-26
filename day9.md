第 9 天：随机森林 Random Forest

第 8 天你学习了决策树。决策树的优点是：

直观、可解释、能可视化、不太需要标准化。

但单棵决策树有一个明显缺点：

容易过拟合，而且对数据变化比较敏感。

今天学习的 随机森林 Random Forest，就是为了解决单棵决策树不稳定的问题。

一、今日学习目标

今天需要掌握 8 个内容：

理解什么是集成学习；
理解 Bagging 思想；
理解随机森林和决策树的关系；
理解随机森林为什么更稳定；
理解 n_estimators 参数；
理解 max_depth 参数；
掌握随机森林特征重要性；
对比决策树和随机森林的效果。
二、先回顾：单棵决策树的问题

决策树是通过一系列规则完成分类。

例如：

petal length <= 2.45？
    是 -> setosa
    否 -> petal width <= 1.75？
        是 -> versicolor
        否 -> virginica

这很直观。

但是单棵决策树有两个问题。

1. 容易过拟合

如果不限制树的深度，决策树可能一直分裂，直到把训练集几乎完全分开。

表现为：

训练集准确率很高；
测试集准确率不一定高。

例如：

训练集 Accuracy = 1.0000
测试集 Accuracy = 0.8500

这说明模型把训练集中的细节甚至噪声都记住了。

2. 对数据变化敏感

决策树很容易因为少量数据变化而改变结构。

比如训练集稍微变动一下，树的第一个分裂条件可能就变了。

这说明单棵树的稳定性不够强。

三、什么是集成学习？

集成学习的核心思想是：

不要只相信一个模型，而是让多个模型一起做决定。

就像答辩评分一样。

如果只有一个老师评分，可能会受个人偏好影响。

如果多个老师一起评分，再取平均分或者投票，结果通常更稳定。

机器学习中也是一样：

单个模型可能不稳定；
多个模型组合起来，整体效果往往更可靠。

随机森林就是一种典型的集成学习方法。

四、随机森林是什么？

随机森林可以理解为：

由很多棵决策树组成的森林。

每一棵树都会给出自己的预测结果。

最后：

分类任务：多棵树投票，哪个类别票数最多，就预测哪个类别；
回归任务：多棵树预测数值，然后取平均值。

例如分类任务中有 5 棵树：

决策树	预测结果
第 1 棵树	setosa
第 2 棵树	versicolor
第 3 棵树	setosa
第 4 棵树	setosa
第 5 棵树	virginica

最终投票结果：

setosa：3 票
versicolor：1 票
virginica：1 票

所以随机森林最终预测：

setosa
五、随机森林为什么叫“随机”？

随机森林有两个重要的随机性。

1. 数据随机

每棵树训练时，不一定使用全部训练数据，而是从训练集中随机抽样。
这种方法叫：
Bootstrap Sampling
也就是：
有放回抽样。

通俗理解：

从原始训练集中随机抽样，组成一份新的训练数据；
每棵树拿到的数据不完全一样；
所以每棵树学到的规则也不完全一样。
2. 特征随机

每棵树在分裂节点时，不一定从全部特征中选择最优特征，而是从一部分随机特征中选择。
例如原始数据有 30 个特征。

单棵决策树每次分裂可能从 30 个特征中选最优。
随机森林中，每棵树每次分裂可能只从其中一部分特征中选。
这样做的好处是：

让每棵树之间差异更大；
避免所有树都长得太像；
整体投票结果更稳定。

六、Bagging 思想
随机森林属于 Bagging 思想。
Bagging 全称是：
Bootstrap Aggregating
可以拆成两部分理解：

Bootstrap：随机有放回抽样；
Aggregating：聚合多个模型结果。

所以 Bagging 的流程是：

第 1 步：从训练集中随机抽样，生成多份子训练集；
第 2 步：每份子训练集训练一个模型；
第 3 步：多个模型一起预测；
第 4 步：分类取投票结果，回归取平均结果。

随机森林就是：

Bagging + 决策树 + 特征随机选择
七、随机森林为什么比单棵树稳定？

单棵决策树就像一个老师评分。

随机森林像多个老师一起评分。

如果某一棵树因为抽到的数据有偏差而判断错了，其他树可能会纠正它。

所以随机森林通常有这些优势：

比单棵决策树更稳定；
更不容易过拟合；
泛化能力通常更好；
仍然可以输出特征重要性；
对标准化不敏感。
八、随机森林常见参数
1. n_estimators
n_estimators=100

表示森林里有多少棵树。

例如：

n_estimators=10：10 棵树
n_estimators=100：100 棵树
n_estimators=300：300 棵树

一般来说：

树越多，模型越稳定；
但训练和预测时间也会增加。

常用初始值：

n_estimators=100
2. max_depth
max_depth=3

表示每棵树的最大深度。

和决策树一样，max_depth 用来控制模型复杂度。

如果不设置：

max_depth=None

树会尽可能生长，直到满足停止条件。

3. random_state
random_state=42

固定随机种子，保证每次运行结果尽量一致。

随机森林本身有随机抽样和随机特征选择，所以固定随机种子很重要。

4. max_features
max_features="sqrt"

表示每次节点分裂时，随机考虑多少个特征。

分类任务中，随机森林默认常用 sqrt，也就是：

每次分裂大约随机考虑 sqrt(总特征数) 个特征。
九、今日案例：对比决策树和随机森林

今天仍然使用 Iris 鸢尾花数据集。

任务是：

根据花萼长度、花萼宽度、花瓣长度、花瓣宽度，判断鸢尾花类别。

我们会训练两个模型：

DecisionTreeClassifier
RandomForestClassifier

然后对比：

训练集准确率；
测试集准确率；
分类报告；
混淆矩阵；
特征重要性。
十、今日完整代码

新建文件：

day09_random_forest.py

代码如下：

"""
第 9 天：随机森林 Random Forest。

本案例使用 sklearn 自带的 Iris 鸢尾花数据集，
对比单棵决策树和随机森林的分类效果。

核心目标：
1. 理解随机森林是多棵决策树的集成
2. 理解 Bagging 思想
3. 理解 n_estimators 参数
4. 对比决策树和随机森林的训练集、测试集效果
5. 查看随机森林的特征重要性
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class RandomForestDemo:
    """
    随机森林分类演示类。

    :param test_size: 测试集比例
    :param random_state: 随机种子，保证实验可复现
    :param n_estimators: 随机森林中决策树的数量
    :param max_depth: 每棵决策树的最大深度
    :return: 无返回值
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 3
    ):
        """
        初始化随机森林分类演示类。

        :param test_size: 测试集比例，例如 0.2 表示 20% 数据作为测试集
        :param random_state: 随机种子，用于保证每次运行结果一致
        :param n_estimators: 随机森林中树的数量
        :param max_depth: 每棵树的最大深度
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self.dataset = None
        self.feature_names = None
        self.target_names = None

        self.X = None
        self.y = None
        self.df = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.decision_tree_model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state
        )

        self.random_forest_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

        self.tree_pred = None
        self.forest_pred = None

    def load_data(self):
        """
        加载 Iris 鸢尾花数据集。

        :return: 无返回值
        """
        self.dataset = load_iris()

        self.X = self.dataset.data
        self.y = self.dataset.target

        self.feature_names = self.dataset.feature_names
        self.target_names = self.dataset.target_names

    def build_dataframe(self):
        """
        构造 DataFrame，方便查看数据。

        :return: 鸢尾花数据 DataFrame
        """
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df["label_id"] = self.y
        self.df["label_name"] = self.df["label_id"].apply(
            lambda label_id: self.target_names[label_id]
        )

        return self.df

    def show_data_info(self):
        """
        打印数据集基本信息。

        :return: 无返回值
        """
        print("========== 数据集基本信息 ==========")
        print("特征名称：", self.feature_names)
        print("类别名称：", self.target_names)
        print("特征数据形状：", self.X.shape)
        print("标签数据形状：", self.y.shape)

        print("\n========== 类别数量统计 ==========")
        print(self.df["label_name"].value_counts())

    def split_data(self):
        """
        划分训练集和测试集。

        :return: 无返回值
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )

        print("\n========== 数据划分结果 ==========")
        print("训练集特征形状：", self.X_train.shape)
        print("测试集特征形状：", self.X_test.shape)
        print("训练集标签形状：", self.y_train.shape)
        print("测试集标签形状：", self.y_test.shape)

    def train_models(self):
        """
        分别训练单棵决策树和随机森林模型。

        :return: 无返回值
        """
        self.decision_tree_model.fit(self.X_train, self.y_train)
        self.random_forest_model.fit(self.X_train, self.y_train)

        print("\n模型训练完成。")
        print(f"决策树 max_depth：{self.max_depth}")
        print(f"随机森林 n_estimators：{self.n_estimators}")
        print(f"随机森林 max_depth：{self.max_depth}")

    def predict_test_data(self):
        """
        使用测试集分别预测。

        :return: 无返回值
        """
        self.tree_pred = self.decision_tree_model.predict(self.X_test)
        self.forest_pred = self.random_forest_model.predict(self.X_test)

    def evaluate_single_model(self, model_name: str, y_pred: np.ndarray):
        """
        评估单个模型。

        :param model_name: 模型名称
        :param y_pred: 模型预测结果
        :return: 无返回值
        """
        accuracy = accuracy_score(self.y_test, y_pred)

        print(f"\n========== {model_name} 模型评估 ==========")
        print(f"测试集 Accuracy：{accuracy:.4f}")

        print("\n混淆矩阵：")
        print(confusion_matrix(self.y_test, y_pred))

        print("\n分类报告：")
        print(classification_report(
            self.y_test,
            y_pred,
            target_names=self.target_names
        ))

    def compare_train_test_accuracy(self):
        """
        对比决策树和随机森林在训练集、测试集上的准确率。

        :return: 对比结果 DataFrame
        """
        tree_train_pred = self.decision_tree_model.predict(self.X_train)
        tree_test_pred = self.tree_pred

        forest_train_pred = self.random_forest_model.predict(self.X_train)
        forest_test_pred = self.forest_pred

        results = [
            {
                "model": "DecisionTree",
                "train_accuracy": accuracy_score(self.y_train, tree_train_pred),
                "test_accuracy": accuracy_score(self.y_test, tree_test_pred)
            },
            {
                "model": "RandomForest",
                "train_accuracy": accuracy_score(self.y_train, forest_train_pred),
                "test_accuracy": accuracy_score(self.y_test, forest_test_pred)
            }
        ]

        result_df = pd.DataFrame(results)

        print("\n========== 决策树 vs 随机森林 准确率对比 ==========")
        print(result_df)

        return result_df

    def show_feature_importance(self):
        """
        输出随机森林特征重要性。

        :return: 特征重要性 DataFrame
        """
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.random_forest_model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="importance",
            ascending=False
        )

        print("\n========== 随机森林特征重要性 ==========")
        print(importance_df)

        return importance_df

    def plot_feature_importance(self, importance_df: pd.DataFrame):
        """
        绘制随机森林特征重要性柱状图。

        :param importance_df: 特征重要性 DataFrame
        :return: 无返回值
        """
        plt.figure(figsize=(9, 5))
        plt.barh(
            importance_df["feature"],
            importance_df["importance"]
        )
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Random Forest Feature Importance")
        plt.gca().invert_yaxis()
        plt.show()

    def predict_single_sample(self, sample: np.ndarray):
        """
        使用随机森林预测单条样本。

        :param sample: 单条样本，格式为 [[花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度]]
        :return: 预测类别 ID 和类别名称
        """
        pred_id = self.random_forest_model.predict(sample)[0]
        pred_name = self.target_names[pred_id]

        return pred_id, pred_name

    def run(self):
        """
        运行完整随机森林流程。

        :return: 无返回值
        """
        print("========== 第 9 天：随机森林 Random Forest ==========")

        self.load_data()
        self.build_dataframe()
        self.show_data_info()

        self.split_data()
        self.train_models()
        self.predict_test_data()

        self.evaluate_single_model(
            model_name="单棵决策树",
            y_pred=self.tree_pred
        )

        self.evaluate_single_model(
            model_name="随机森林",
            y_pred=self.forest_pred
        )

        self.compare_train_test_accuracy()

        importance_df = self.show_feature_importance()
        self.plot_feature_importance(importance_df)

        sample = np.array([[6.0, 3.0, 4.8, 1.8]])
        pred_id, pred_name = self.predict_single_sample(sample)

        print("\n========== 单条样本预测 ==========")
        print("输入样本：", sample)
        print("随机森林预测类别 ID：", pred_id)
        print("随机森林预测类别名称：", pred_name)


class RandomForestParameterExperiment:
    """
    随机森林参数对比实验类。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化随机森林参数对比实验类。

        :param test_size: 测试集比例
        :param random_state: 随机种子
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state

        self.dataset = load_iris()
        self.X = self.dataset.data
        self.y = self.dataset.target

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.result_df = None

    def prepare_data(self):
        """
        准备训练集和测试集。

        :return: 无返回值
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )

    def compare_n_estimators(self, estimator_values: list):
        """
        比较不同 n_estimators 下随机森林的训练集和测试集准确率。

        :param estimator_values: 树数量列表，例如 [1, 5, 10, 50, 100]
        :return: 实验结果 DataFrame
        """
        results = []

        for n_estimators in estimator_values:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=3,
                random_state=self.random_state
            )

            model.fit(self.X_train, self.y_train)

            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)

            train_accuracy = accuracy_score(self.y_train, train_pred)
            test_accuracy = accuracy_score(self.y_test, test_pred)

            results.append({
                "n_estimators": n_estimators,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy
            })

            print(
                f"n_estimators={n_estimators}，"
                f"训练集准确率={train_accuracy:.4f}，"
                f"测试集准确率={test_accuracy:.4f}"
            )

        self.result_df = pd.DataFrame(results)
        return self.result_df

    def plot_estimator_accuracy_curve(self):
        """
        绘制 n_estimators 与准确率关系图。

        :return: 无返回值
        """
        plt.figure(figsize=(8, 5))

        plt.plot(
            self.result_df["n_estimators"],
            self.result_df["train_accuracy"],
            marker="o",
            label="Train Accuracy"
        )

        plt.plot(
            self.result_df["n_estimators"],
            self.result_df["test_accuracy"],
            marker="o",
            label="Test Accuracy"
        )

        plt.xlabel("n_estimators")
        plt.ylabel("Accuracy")
        plt.title("Random Forest n_estimators vs Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        """
        运行随机森林参数对比实验。

        :return: 无返回值
        """
        print("\n========== 随机森林 n_estimators 对比实验 ==========")

        self.prepare_data()

        estimator_values = [1, 3, 5, 10, 30, 50, 100, 200]
        self.compare_n_estimators(estimator_values=estimator_values)

        print("\n========== n_estimators 对比结果 ==========")
        print(self.result_df)

        self.plot_estimator_accuracy_curve()


def main():
    """
    主函数。

    :return: 无返回值
    """
    demo = RandomForestDemo(
        test_size=0.2,
        random_state=42,
        n_estimators=100,
        max_depth=3
    )
    demo.run()

    experiment = RandomForestParameterExperiment(
        test_size=0.2,
        random_state=42
    )
    experiment.run()


if __name__ == "__main__":
    main()
十一、代码重点解释
1. 创建单棵决策树
self.decision_tree_model = DecisionTreeClassifier(
    max_depth=self.max_depth,
    random_state=self.random_state
)

这就是第 8 天学过的模型。

它只有一棵树。

2. 创建随机森林
self.random_forest_model = RandomForestClassifier(
    n_estimators=self.n_estimators,
    max_depth=self.max_depth,
    random_state=self.random_state
)

这里表示：

创建一个随机森林；
森林里有 n_estimators 棵决策树；
每棵树最大深度为 max_depth。
3. 训练模型
self.decision_tree_model.fit(self.X_train, self.y_train)
self.random_forest_model.fit(self.X_train, self.y_train)

这两行分别训练：

单棵决策树；
随机森林。
4. 对比训练集和测试集效果
train_accuracy = accuracy_score(self.y_train, train_pred)
test_accuracy = accuracy_score(self.y_test, test_pred)

这里要重点看：

训练集准确率和测试集准确率之间的差距。

如果训练集远高于测试集，说明可能过拟合。

5. 查看特征重要性
self.random_forest_model.feature_importances_

随机森林可以输出每个特征的重要性。

例如：

petal width (cm)     0.48
petal length (cm)    0.43
sepal length (cm)    0.07
sepal width (cm)     0.02

可以理解为：

花瓣宽度和花瓣长度对分类贡献最大。
十二、运行后重点观察什么？

运行代码后，重点观察 4 个地方。

1. 决策树和随机森林的准确率对比

你会看到类似结果：

          model  train_accuracy  test_accuracy
0  DecisionTree        0.966667       0.966667
1  RandomForest        0.966667       0.966667

Iris 数据集比较简单，所以两者可能差别不大。

但是在复杂数据中，随机森林通常比单棵树更稳定。

2. 混淆矩阵

随机森林和决策树都会输出混淆矩阵。

你要看：

哪些类别容易被混淆；
错误主要发生在哪些类别之间。

对于 Iris 数据集，常见错误是：

versicolor 和 virginica 容易混淆；
setosa 通常很容易区分。
3. 特征重要性

重点看：

随机森林认为哪些特征最重要？

通常你会发现：

petal length
petal width

这两个花瓣相关特征最重要。

4. n_estimators 对比实验

你会看到不同树数量下的准确率变化。

例如：

n_estimators=1，测试集准确率=0.9333
n_estimators=10，测试集准确率=0.9667
n_estimators=100，测试集准确率=0.9667

你要理解：

树的数量增加，模型通常更稳定；
但准确率不一定无限提升；
树越多，计算成本越高。
十三、随机森林和决策树的区别
对比项	决策树	随机森林
模型数量	一棵树	多棵树
核心思想	单棵树规则判断	多棵树投票
稳定性	较弱	较强
过拟合风险	较高	相对较低
可解释性	很强	比单棵树弱
训练速度	快	较慢
预测速度	快	较慢
特征重要性	支持	支持
是否需要标准化	通常不需要	通常不需要

一句话总结：

决策树是一个老师判断；
随机森林是多个老师投票。
十四、随机森林为什么通常不需要标准化？

KNN 需要标准化，因为它依赖距离。

逻辑回归通常建议标准化，因为它要优化参数，特征尺度会影响收敛。

但随机森林和决策树类似，主要根据特征阈值进行切分。

例如：

petal length <= 2.45
area <= 100
income <= 5000

所以它对特征尺度不敏感。

因此：

随机森林通常不需要 StandardScaler。
十五、随机森林适合什么场景？

随机森林适合很多表格数据任务。

例如：

用户是否流失；
订单是否异常；
客户是否会购买；
设备是否故障；
学生成绩等级预测；
商品销量等级预测；
贷款是否违约；
用户分层分类。

它的优势是：

效果稳定；
调参相对简单；
能处理非线性关系；
能输出特征重要性；
对标准化不敏感。

它的不足是：

模型比单棵树更难解释；
树很多时模型体积较大；
预测速度比单棵树慢；
不能像线性模型那样直接解释每个权重含义。
十六、随机森林常见调参思路
1. 先调 n_estimators
n_estimators=100

常用范围：

50、100、200、300

数据量不大时，100 通常够用。

2. 再调 max_depth
max_depth=3
max_depth=5
max_depth=10
max_depth=None

如果模型欠拟合，可以适当增大 max_depth。

如果模型过拟合，可以减小 max_depth。

3. 调 min_samples_leaf
min_samples_leaf=1
min_samples_leaf=2
min_samples_leaf=5

这个参数可以限制每个叶子节点最少样本数。

增大它可以减少过拟合。

4. 调 max_features
max_features="sqrt"
max_features="log2"

这个参数控制每次分裂时考虑多少个特征。

特征很多时，max_features 对随机森林效果有影响。

十七、随机森林的核心代码模板

以后使用随机森林，常见模板是：

"""
随机森林分类模型基础模板。
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# 1. 准备特征和标签
X = ...
y = ...

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. 创建随机森林模型
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 预测测试集
y_pred = model.predict(X_test)

# 6. 评估模型
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

# 7. 查看特征重要性
print(model.feature_importances_)
十八、今日练习
练习 1：解释随机森林

请用自己的话解释：

随机森林为什么比单棵决策树更稳定？

参考方向：

随机森林由很多棵决策树组成；
每棵树看到的数据和特征不完全一样；
最终通过投票得到结果；
单棵树的错误可能被其他树抵消。
练习 2：修改树的数量

把代码中的：

n_estimators=100

分别改成：

n_estimators=1
n_estimators=5
n_estimators=10
n_estimators=200

观察：

训练集准确率是否变化？
测试集准确率是否变化？
模型运行时间是否变化？
练习 3：修改最大深度

把代码中的：

max_depth=3

分别改成：

max_depth=1
max_depth=2
max_depth=5
max_depth=None

观察：

训练集准确率是否升高？
测试集准确率是否同步升高？
是否出现训练集很高但测试集没有提升的情况？
练习 4：查看特征重要性

运行代码后，找到：

随机森林特征重要性

回答：

哪个特征最重要？
哪个特征最不重要？
花瓣特征和花萼特征哪个整体更重要？
练习 5：替换数据集

把 Iris 数据集换成乳腺癌数据集：

from sklearn.datasets import load_breast_cancer

self.dataset = load_breast_cancer()

然后重新训练随机森林。

观察：

特征数量是多少？
类别名称是什么？
随机森林准确率是多少？
哪些特征重要性最高？
十九、今日学习检查表
检查项	是否掌握
能理解随机森林是多棵决策树	
能理解集成学习思想	
能理解 Bagging 思想	
能理解 Bootstrap 抽样	
能理解多棵树投票机制	
能理解 n_estimators 参数	
能理解 max_depth 参数	
能使用 RandomForestClassifier	
能对比决策树和随机森林	
能查看随机森林特征重要性	
二十、今日总结

第 9 天最重要的是这句话：

随机森林就是很多棵决策树组成的集成模型，通过随机抽样、随机特征选择和多树投票，提高模型稳定性和泛化能力。

你现在应该能理解：

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

背后的含义是：

创建 100 棵决策树；
每棵树基于随机数据和随机特征学习规则；
预测时每棵树投票；
最终票数最多的类别作为预测结果。

今天还要记住：

决策树：一棵树，容易解释，但稳定性较弱；
随机森林：多棵树，稳定性更强，但解释性比单棵树弱。