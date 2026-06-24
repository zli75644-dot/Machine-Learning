第 7 天：KNN 最近邻算法

今天学习一个非常直观的分类算法：KNN，K-Nearest Neighbors，K 近邻算法。

KNN 的思想非常简单：

一个新样本属于哪一类，要看它周围最近的 K 个样本大多数属于哪一类。

你可以把 KNN 理解成：

近朱者赤，近墨者黑。

也就是说：

一个样本和谁最像，就更可能属于谁那一类。
一、今日学习目标

今天需要掌握 7 个内容：

理解 KNN 的核心思想；
理解什么是距离度量；
理解 K 值的含义；
理解 K 值太小和太大的问题；
理解为什么 KNN 需要标准化；
使用 sklearn 实现 KNN 分类；
观察不同 K 值对模型准确率的影响。
二、KNN 是什么？

KNN 全称是：

K-Nearest Neighbors

中文叫：

K 近邻算法

它是一种常见的监督学习算法，可以用于：

分类任务
回归任务

但入门阶段我们主要用它做分类。

三、KNN 的核心思想

假设我们有一些已经知道类别的数据。

例如根据身高和体重判断一个人更像篮球运动员还是普通人：

身高	体重	类别
190	    85	篮球运动员
195	    90	篮球运动员
188 	82	篮球运动员
165	    55	普通人
170	    60	普通人
172 	65	普通人

现在来了一个新人：

身高 = 187
体重 = 80

KNN 会做下面几件事：

第 1 步：计算这个新人和所有历史样本的距离；
第 2 步：找到距离最近的 K 个样本；
第 3 步：看这 K 个样本中哪个类别最多；
第 4 步：把新人预测为最多的那个类别。

如果设置：

K = 3

并且最近的 3 个样本都是篮球运动员，那么模型就预测：

这个新人更可能是篮球运动员。
四、什么是距离？

KNN 的核心是“近”。

但是“近”在机器学习中需要用数学方式表示，也就是：

距离

最常见的是欧氏距离。

例如一个样本有两个特征：

x1 = 身高
x2 = 体重

两个样本分别是：

A = (170, 60)
B = (180, 75)

它们之间的欧氏距离是：

代入数据：

d(A, B) = sqrt((170 - 180)^2 + (60 - 75)^2)
        = sqrt(100 + 225)
        = sqrt(325)
        ≈ 18.03

距离越小，说明两个样本越相似。

五、K 值是什么意思？

KNN 里的 K 表示：

选择最近的几个邻居来投票。

例如：

K = 1：只看最近的 1 个邻居；
K = 3：看最近的 3 个邻居；
K = 5：看最近的 5 个邻居；
K = 7：看最近的 7 个邻居。

假设 K = 5，最近的 5 个邻居类别如下：

类别 A
类别 A
类别 B
类别 A
类别 B

那么：

类别 A 有 3 票
类别 B 有 2 票

所以最终预测为：

类别 A
六、K 值太小和太大的问题
1. K 值太小

例如：

K = 1

模型只看最近的一个样本。

优点是：

对局部数据非常敏感。

缺点是：

容易受到异常点影响。

比如刚好最近的那个样本是噪声数据，模型就可能预测错。

这时候容易出现：

过拟合

也就是模型太依赖训练数据中的细节，对新数据泛化不好。

2. K 值太大

例如：

K = 99

模型看太多邻居。

优点是：

结果更平滑，不容易被单个异常点影响。

缺点是：

可能把距离很远、不太相关的样本也算进来。

这时候容易出现：

欠拟合

也就是模型太粗糙，分类边界不够清晰。

3. K 值怎么选？

常见做法是：

尝试多个 K 值，然后用测试集或交叉验证选择效果最好的 K。

例如测试：

K = 1、3、5、7、9、11、13、15

然后观察哪个 K 的准确率最高。

七、为什么 KNN 需要标准化？

KNN 非常依赖距离。

如果不同特征的数值范围差异很大，距离会被数值大的特征主导。

例如有两个特征：

年龄：18 到 60
收入：3000 到 50000

如果直接计算距离，收入的数值范围远远大于年龄。

那么模型计算距离时，收入会占主导地位，年龄的影响会变得很小。

所以 KNN 一般需要标准化：

让不同特征处在相近的尺度上。

常用方法：

StandardScaler()

它会把数据转换成：

均值约为 0
标准差约为 1

这样每个特征在距离计算中会更加公平。

八、KNN 的优缺点
1. 优点

KNN 的优点：

原理简单；
容易理解；
不需要复杂训练过程；
适合小规模数据；
可以用于分类和回归。
2. 缺点

KNN 的缺点：

预测时计算量较大；
对特征尺度敏感；
对噪声和异常值敏感；
高维数据中效果可能下降；
需要选择合适的 K 值。

KNN 和逻辑回归、决策树不同。

很多模型训练时比较费时间，但预测较快。

KNN 是：

训练几乎很简单；
预测时要计算新样本和大量训练样本的距离。

所以数据量很大时，KNN 预测会比较慢。

九、今日案例：鸢尾花分类

今天继续使用 Iris 鸢尾花数据集。

任务是：

根据花萼长度、花萼宽度、花瓣长度、花瓣宽度，判断鸢尾花类别。

特征 X：

sepal length
sepal width
petal length
petal width

标签 y：

setosa
versicolor
virginica

这是一个三分类任务。

十、今日完整代码

新建文件：

day07_knn_classification.py

代码如下：

"""
第 7 天：KNN 最近邻算法。

本案例使用 sklearn 自带的 Iris 鸢尾花数据集，
演示 KNN 分类算法的完整流程。

核心目标：
1. 理解 KNN 的基本思想
2. 理解 K 值对模型效果的影响
3. 理解标准化对 KNN 的重要性
4. 对比不同 K 值下的准确率
5. 使用 matplotlib 绘制 K 值与准确率变化图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class IrisKNNClassifier:
    """
    鸢尾花 KNN 分类器。

    :param test_size: 测试集比例
    :param random_state: 随机种子，保证每次划分数据结果一致
    :param n_neighbors: KNN 算法中的 K 值
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42, n_neighbors: int = 3):
        """
        初始化鸢尾花 KNN 分类器。

        :param test_size: 测试集比例，例如 0.2 表示 20% 数据作为测试集
        :param random_state: 随机种子，用于保证实验结果可复现
        :param n_neighbors: KNN 算法中的 K 值
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state
        self.n_neighbors = n_neighbors

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

        self.scaler = StandardScaler()
        self.X_train_scaled = None
        self.X_test_scaled = None

        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.y_pred = None

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

        print("\n========== 前 5 行数据 ==========")
        print(self.df.head())

        print("\n========== 类别数量统计 ==========")
        print(self.df["label_name"].value_counts())

    def split_data(self):
        """
        划分训练集和测试集。

        stratify=self.y 表示按照标签比例分层抽样，
        保证训练集和测试集中各类别比例尽量一致。

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

    def scale_features(self):
        """
        对特征进行标准化。

        KNN 是基于距离的算法，不同特征的数值尺度会影响距离计算。
        因此，KNN 通常需要进行标准化。

        注意：
        1. 训练集使用 fit_transform
        2. 测试集使用 transform

        :return: 无返回值
        """
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train_model(self):
        """
        训练 KNN 分类模型。

        严格来说，KNN 的训练过程主要是保存训练数据。
        真正的距离计算发生在预测阶段。

        :return: 无返回值
        """
        self.model.fit(self.X_train_scaled, self.y_train)
        print(f"\nKNN 模型训练完成，当前 K 值为：{self.n_neighbors}")

    def predict_test_data(self):
        """
        使用测试集进行预测。

        :return: 测试集预测结果
        """
        self.y_pred = self.model.predict(self.X_test_scaled)
        return self.y_pred

    def evaluate_model(self):
        """
        评估 KNN 分类模型。

        :return: 无返回值
        """
        accuracy = accuracy_score(self.y_test, self.y_pred)

        print("\n========== 模型评估结果 ==========")
        print(f"Accuracy 准确率：{accuracy:.4f}")

        print("\n========== 混淆矩阵 ==========")
        print(confusion_matrix(self.y_test, self.y_pred))

        print("\n========== 分类报告 ==========")
        print(classification_report(
            self.y_test,
            self.y_pred,
            target_names=self.target_names
        ))

    def show_prediction_detail(self, rows: int = 10):
        """
        展示部分测试样本的真实类别和预测类别。

        :param rows: 展示前多少行
        :return: 无返回值
        """
        result_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        result_df["真实类别ID"] = self.y_test
        result_df["预测类别ID"] = self.y_pred

        result_df["真实类别名称"] = result_df["真实类别ID"].apply(
            lambda label_id: self.target_names[label_id]
        )

        result_df["预测类别名称"] = result_df["预测类别ID"].apply(
            lambda label_id: self.target_names[label_id]
        )

        result_df["是否预测正确"] = result_df["真实类别ID"] == result_df["预测类别ID"]

        print("\n========== 部分预测结果 ==========")
        print(result_df.head(rows))

    def predict_single_sample(self, sample: np.ndarray):
        """
        预测单条鸢尾花样本。

        :param sample: 单条样本，格式为 [[花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度]]
        :return: 预测类别 ID 和类别名称
        """
        sample_scaled = self.scaler.transform(sample)

        pred_id = self.model.predict(sample_scaled)[0]
        pred_name = self.target_names[pred_id]

        return pred_id, pred_name

    def run(self):
        """
        运行完整 KNN 分类流程。

        :return: 无返回值
        """
        print("========== 第 7 天：KNN 最近邻算法 ==========")

        self.load_data()
        self.build_dataframe()
        self.show_data_info()

        self.split_data()
        self.scale_features()

        self.train_model()
        self.predict_test_data()
        self.evaluate_model()
        self.show_prediction_detail(rows=10)

        sample = np.array([[6.0, 3.0, 4.8, 1.8]])
        pred_id, pred_name = self.predict_single_sample(sample)

        print("\n========== 单条样本预测 ==========")
        print("输入样本：", sample)
        print("预测类别ID：", pred_id)
        print("预测类别名称：", pred_name)


class KValueComparisonExperiment:
    """
    K 值对比实验类。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化 K 值对比实验类。

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

        self.scaler = StandardScaler()
        self.X_train_scaled = None
        self.X_test_scaled = None

        self.result_df = None

    def prepare_data(self):
        """
        准备训练集和测试集，并完成标准化。

        :return: 无返回值
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def compare_different_k_values(self, k_values: list):
        """
        比较不同 K 值下的模型准确率。

        :param k_values: K 值列表，例如 [1, 3, 5, 7, 9]
        :return: 实验结果 DataFrame
        """
        results = []

        for k in k_values:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(self.X_train_scaled, self.y_train)

            y_pred = model.predict(self.X_test_scaled)
            accuracy = accuracy_score(self.y_test, y_pred)

            results.append({
                "k": k,
                "accuracy": accuracy
            })

            print(f"K = {k}，Accuracy = {accuracy:.4f}")

        self.result_df = pd.DataFrame(results)
        return self.result_df

    def plot_k_accuracy_curve(self):
        """
        绘制 K 值和准确率的关系图。

        :return: 无返回值
        """
        plt.figure(figsize=(8, 5))
        plt.plot(
            self.result_df["k"],
            self.result_df["accuracy"],
            marker="o"
        )

        plt.xlabel("K Value")
        plt.ylabel("Accuracy")
        plt.title("K Value vs Accuracy")
        plt.xticks(self.result_df["k"])
        plt.grid(True)
        plt.show()

    def run(self):
        """
        运行 K 值对比实验。

        :return: 无返回值
        """
        print("\n========== K 值对比实验 ==========")

        self.prepare_data()

        k_values = [1, 3, 5, 7, 9, 11, 13, 15]
        self.compare_different_k_values(k_values=k_values)

        print("\n========== K 值对比结果 ==========")
        print(self.result_df)

        self.plot_k_accuracy_curve()


def main():
    """
    主函数。

    :return: 无返回值
    """
    knn_demo = IrisKNNClassifier(
        test_size=0.2,
        random_state=42,
        n_neighbors=3
    )
    knn_demo.run()

    experiment = KValueComparisonExperiment(
        test_size=0.2,
        random_state=42
    )
    experiment.run()


if __name__ == "__main__":
    main()
十一、代码重点解释
1. 创建 KNN 模型
self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

这里的：

n_neighbors=3

表示：

K = 3

也就是预测新样本时，看最近的 3 个邻居。

2. 标准化特征
self.X_train_scaled = self.scaler.fit_transform(self.X_train)
self.X_test_scaled = self.scaler.transform(self.X_test)

这一步对 KNN 很重要。

因为 KNN 需要计算距离，如果不标准化，数值范围大的特征会影响更大。

注意：

训练集：fit_transform
测试集：transform

不能对测试集单独 fit_transform，否则会造成数据处理规则不一致，也可能导致数据泄露。

3. 训练模型
self.model.fit(self.X_train_scaled, self.y_train)

KNN 的 fit() 和逻辑回归不太一样。

逻辑回归训练时会学习权重 w 和偏置 b。

KNN 训练时主要是：

保存训练数据。

真正复杂的计算在预测阶段发生。

4. 预测模型
self.y_pred = self.model.predict(self.X_test_scaled)

预测时，KNN 会：

计算测试样本和训练样本之间的距离；
找到最近的 K 个样本；
根据这 K 个样本投票；
输出最终类别。
5. 比较不同 K 值
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

这部分代码会依次训练多个 KNN 模型：

model = KNeighborsClassifier(n_neighbors=k)

然后记录不同 K 值下的准确率。

最终画出：

K 值与 Accuracy 的关系图
十二、运行后重点观察什么？

运行代码后，重点观察 3 个部分。

1. 当前 K 值下的模型效果

例如：

KNN 模型训练完成，当前 K 值为：3
Accuracy 准确率：0.9667

说明：

当前使用 K = 3 时，测试集准确率约为 96.67%。
2. 混淆矩阵

你会看到类似：

[[10  0  0]
 [ 0  9  1]
 [ 0  1  9]]

因为 Iris 是三分类任务，所以混淆矩阵是 3 × 3。

含义是：

第 1 行：真实类别是 setosa 的样本预测情况；
第 2 行：真实类别是 versicolor 的样本预测情况；
第 3 行：真实类别是 virginica 的样本预测情况。

对角线上的数字表示预测正确。

非对角线上的数字表示预测错误。

3. 不同 K 值的准确率

你会看到类似：

K = 1，Accuracy = 0.9667
K = 3，Accuracy = 0.9667
K = 5，Accuracy = 1.0000
K = 7，Accuracy = 0.9667

注意：

不是 K 越大越好；
也不是 K 越小越好；
要通过实验选择合适的 K。
十三、手动理解 KNN 的预测过程

假设现在有一个新样本：

sample = [6.0, 3.0, 4.8, 1.8]

模型会把它和训练集中的所有样本计算距离。

然后找到最近的 3 个样本。

假设最近的 3 个样本类别是：

virginica
virginica
versicolor

那么投票结果是：

virginica：2 票
versicolor：1 票

所以最终预测为：

virginica

这就是 KNN 的核心过程。

十四、KNN 和逻辑回归的区别
对比项  	        KNN	                    逻辑回归
核心思想	    看最近邻居投票	        学习权重和偏置
是否显式学习参数	基本不学习 w、b	    学习 w、b
训练速度	        快	                较快
预测速度	       数据量大时慢         快
是否依赖距离	      是            	否
是否需要标准化	    通常需要	        通常建议
可解释性	           直观         	较强
适合数据	    小规模、边界较清晰  	线性可分或近似线性关系

一句话总结：

逻辑回归是学一个分类公式；
KNN 是看新样本附近的邻居怎么投票。
十五、KNN 的一个重要特点：懒惰学习

KNN 有时也叫：

懒惰学习算法

因为它训练时基本不做复杂计算，只是保存数据。

等到预测时才开始计算距离。

所以：

训练阶段：比较轻松；
预测阶段：比较费劲。

这和很多模型不同。

例如逻辑回归：

训练阶段：学习 w 和 b；
预测阶段：直接代入公式，速度很快。
十六、KNN 中常见参数
1. n_neighbors
KNeighborsClassifier(n_neighbors=3)

表示 K 值。

这是最重要的参数。

2. weights
KNeighborsClassifier(n_neighbors=5, weights="uniform")

weights 表示投票方式。

常见取值：

uniform：所有邻居权重相同；
distance：距离越近，权重越大。

例如：

KNeighborsClassifier(n_neighbors=5, weights="distance")

表示：

近的邻居投票影响更大。
3. metric
KNeighborsClassifier(metric="minkowski")

表示距离计算方式。

默认一般是欧氏距离。

你入门阶段先记住：

KNN 核心是距离，距离计算方式会影响结果。
十七、今日练习
练习 1：解释 KNN

请用自己的话解释：

KNN 是如何判断一个新样本类别的？

参考回答方向：

先计算新样本和训练集中所有样本的距离；
找到最近的 K 个邻居；
看这些邻居中哪个类别最多；
把新样本预测为这个类别。
练习 2：修改 K 值

把代码中的：

n_neighbors=3

分别改成：

n_neighbors=1
n_neighbors=5
n_neighbors=7
n_neighbors=9

观察模型准确率是否变化。

练习 3：关闭标准化

尝试把代码中的：

self.scale_features()

以及模型训练和预测时使用的：

self.X_train_scaled
self.X_test_scaled

改成：

self.X_train
self.X_test

观察准确率是否变化。

思考：

为什么 KNN 对标准化敏感？
练习 4：修改单条预测样本

修改代码中的：

sample = np.array([[6.0, 3.0, 4.8, 1.8]])

分别测试：

sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample = np.array([[6.5, 3.0, 5.2, 2.0]])
sample = np.array([[5.8, 2.7, 4.1, 1.0]])

观察预测类别分别是什么。

练习 5：使用 distance 权重

把模型改成：

self.model = KNeighborsClassifier(
    n_neighbors=self.n_neighbors,
    weights="distance"
)

然后重新运行。

观察：

准确率是否变化？
预测结果是否变化？
十八、今日学习检查表
检查项	是否掌握
能理解 KNN 的基本思想	
能解释 K 表示什么	
能理解欧氏距离	
能理解 K 值太小容易过拟合	
能理解 K 值太大容易欠拟合	
能理解 KNN 为什么需要标准化	
能使用 KNeighborsClassifier	
能比较不同 K 值的准确率	
能看懂 K 值和 Accuracy 的关系图	
能独立完成 Iris KNN 分类实验	
十九、今日总结

第 7 天最重要的是这句话：

KNN 是基于距离的投票算法：新样本属于哪一类，取决于它最近的 K 个邻居大多数属于哪一类。

你现在应该能理解：

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

背后的含义：

保存训练数据；
预测时计算距离；
找到最近的 3 个邻居；
根据邻居类别投票；
输出预测结果。

今天还要记住：

K 值太小，容易受噪声影响；
K 值太大，分类边界可能太粗糙；
KNN 依赖距离，所以通常需要标准化。