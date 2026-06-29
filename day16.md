第 16 天学习 PCA 降维：它不是删除某几列特征，而是把原来的多维特征重新组合成更少、更重要的新特征，常用于可视化、降噪和压缩。

已思考 5s
第 16 天：PCA 降维 Principal Component Analysis

前面第 15 天你学习了聚类。聚类时我们有多个特征：

annual_spend
order_count
avg_order_amount
discount_rate
return_rate

如果特征只有 2 个，我们可以直接画二维图观察数据分布。

但真实项目中，特征可能有：

30 个
100 个
1000 个
甚至更多

特征太多时，会带来几个问题：

不好可视化；
训练速度变慢；
特征之间可能有冗余；
噪声特征可能影响模型；
高维空间中距离计算可能变差。

今天学习的 PCA 就是解决这些问题的一种经典方法。

一、今日学习目标

今天需要掌握 8 个内容：

什么是降维；
为什么需要降维；
PCA 的核心思想；
什么是主成分；
什么是解释方差；
PCA 和特征选择的区别；
使用 PCA 将高维数据降到二维进行可视化；
使用 PCA 降维后训练分类模型。
二、什么是降维？

降维就是：

把高维数据转换成低维数据，同时尽量保留原始数据中的主要信息。

例如原来有 30 个特征：

x1, x2, x3, ..., x30

经过 PCA 后，可以变成 2 个新特征：

PC1, PC2

这两个新特征不是原来的某两列，而是原始特征的组合。

例如：

PC1 = 0.3*x1 + 0.2*x2 - 0.1*x3 + ...
PC2 = -0.1*x1 + 0.4*x2 + 0.2*x3 + ...

所以 PCA 不是简单删列，而是重新生成新的特征空间。

三、为什么需要降维？
1. 方便可视化

如果数据有 30 个特征，我们无法直接画 30 维图。

但可以用 PCA 降到二维：

30 维数据 → 2 维数据

然后画散点图，观察类别是否分得开。

2. 减少冗余特征

很多特征之间可能高度相关。

例如房价数据中：

房屋面积
卧室数量
客厅面积
总价

这些字段之间可能有很强关系。

PCA 可以把相关特征压缩成少数几个主成分。

3. 降低噪声

有些特征可能主要是噪声。

PCA 会优先保留数据变化最大的方向，舍弃变化较小的方向。

很多时候，这可以起到一定降噪作用。

4. 提高训练效率

特征数量变少后，模型训练和预测可能更快。

例如：

原始特征：1000 个
PCA 后：50 个

模型需要处理的数据维度就大幅减少了。

四、PCA 的核心思想

PCA 全称是：

Principal Component Analysis

中文叫：

主成分分析

它的核心思想是：

找到数据中变化最大的方向，把这些方向作为新的坐标轴。

例如二维数据中，数据点大致沿着一个斜方向分布。

原始坐标轴可能是：

x 轴
y 轴

但 PCA 会找到一个新的方向：

数据变化最大的方向

这个方向就是第一主成分：

PC1

然后再找一个和 PC1 垂直、并且能解释剩余变化最多的方向：

PC2
五、什么是主成分？

主成分就是 PCA 找到的新特征方向。

1. 第一主成分 PC1

第一主成分是：

能保留最多数据变化信息的方向。

也就是说，数据沿这个方向变化最大。

2. 第二主成分 PC2

第二主成分是：

在和 PC1 垂直的前提下，能保留第二多信息的方向。
3. 第三主成分 PC3

第三主成分继续保留剩余信息。

以此类推。

所以主成分的重要性通常是：

PC1 > PC2 > PC3 > PC4 ...
六、解释方差 Explained Variance

PCA 中有一个非常重要的概念：

解释方差 explained variance

它表示：

每个主成分保留了多少原始数据信息。

例如 PCA 输出：

PC1 解释方差比例：0.44
PC2 解释方差比例：0.19
PC3 解释方差比例：0.09

表示：

PC1 保留了 44% 的信息；
PC2 保留了 19% 的信息；
PC3 保留了 9% 的信息。

如果只保留前两个主成分：

PC1 + PC2 = 0.44 + 0.19 = 0.63

说明二维 PCA 保留了原始数据约 63% 的信息。

七、PCA 的基础公式理解

PCA 的核心可以粗略理解为：

寻找新的方向，让数据投影到这些方向后方差最大。

方差越大，说明数据在这个方向上的区分度越明显。

这里可以先这样理解：

X：原始特征矩阵；
W：PCA 学到的主成分方向；
Z：降维后的新特征矩阵。

也就是说：

PCA 不是直接选择原来的某几列；
PCA 是通过矩阵变换生成新的特征。
八、PCA 和特征选择的区别
对比项	            特征选择	                PCA 降维
是否保留原始字段	是	                            否
做法	            从原始特征中选一部分	把原始特征重新组合成新特征
输出结果	        原始字段子集	        主成分 PC1、PC2
可解释性	        较强	                较弱
是否改变特征含义	不改变	                改变
常见用途	        保留重要字段	        压缩、降噪、可视化

举例：

特征选择：从 30 个字段中选出 5 个原字段。
PCA：把 30 个字段组合成 5 个新主成分。

所以：

特征选择更容易解释；
PCA 更适合压缩和可视化。
九、PCA 为什么通常需要标准化？

PCA 依赖方差。

如果不同特征尺度差异很大，方差大的特征会主导 PCA。

例如：

年龄：18 到 60
收入：3000 到 50000

收入的方差远大于年龄。

如果不标准化，PCA 可能主要围绕收入方向找主成分。

所以 PCA 前通常要做：

StandardScaler()

让每个特征处于相近尺度。

十、今日案例：乳腺癌数据 PCA 可视化与分类

今天使用 sklearn 自带的乳腺癌数据集。

这个数据集有：

569 条样本
30 个特征
2 个类别

任务分两部分：

第一部分：用 PCA 把 30 维数据降到 2 维，然后画二维散点图；
第二部分：比较原始特征训练模型和 PCA 特征训练模型的效果。
十一、今日完整代码

新建文件：

day16_pca_dimensionality_reduction.py

代码如下：

"""
第 16 天：PCA 降维 Principal Component Analysis。

本案例使用 sklearn 自带的乳腺癌数据集，演示 PCA 的完整流程：

1. 加载高维数据
2. 标准化特征
3. 使用 PCA 降到二维
4. 查看解释方差比例
5. 绘制二维 PCA 可视化图
6. 比较原始特征模型和 PCA 特征模型的分类效果
7. 观察不同 n_components 下保留的信息比例

核心目标：
理解 PCA 不是简单删除特征，而是把原始特征重新组合成新的主成分。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class BreastCancerDataLoader:
    """
    乳腺癌数据加载类。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化乳腺癌数据加载类。

        :param test_size: 测试集比例
        :param random_state: 随机种子
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state

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

    def load_data(self):
        """
        加载 sklearn 自带的乳腺癌数据集。

        :return: 无返回值
        """
        self.dataset = load_breast_cancer()

        self.X = self.dataset.data
        self.y = self.dataset.target

        self.feature_names = self.dataset.feature_names
        self.target_names = self.dataset.target_names

    def build_dataframe(self):
        """
        构造 DataFrame，方便查看数据。

        :return: 数据 DataFrame
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
        print("特征数据形状：", self.X.shape)
        print("标签数据形状：", self.y.shape)
        print("特征数量：", len(self.feature_names))
        print("类别名称：", self.target_names)

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

        print("\n========== 数据集划分结果 ==========")
        print("训练集特征形状：", self.X_train.shape)
        print("测试集特征形状：", self.X_test.shape)
        print("训练集标签形状：", self.y_train.shape)
        print("测试集标签形状：", self.y_test.shape)


class PCAVisualizer:
    """
    PCA 可视化类。

    :param n_components: PCA 降维后的维度数量
    :return: 无返回值
    """

    def __init__(self, n_components: int = 2):
        """
        初始化 PCA 可视化类。

        :param n_components: PCA 主成分数量
        :return: 无返回值
        """
        self.n_components = n_components

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)

        self.X_scaled = None
        self.X_pca = None
        self.pca_df = None

    def fit_transform(self, X: np.ndarray, y: np.ndarray, target_names: np.ndarray) -> pd.DataFrame:
        """
        对全部数据进行标准化和 PCA 降维，用于可视化。

        注意：
        这里为了画图演示，对全部数据做 PCA。
        如果用于建模评估，应当在训练集 fit，在测试集 transform。

        :param X: 原始特征
        :param y: 标签
        :param target_names: 类别名称
        :return: PCA 后的 DataFrame
        """
        self.X_scaled = self.scaler.fit_transform(X)
        self.X_pca = self.pca.fit_transform(self.X_scaled)

        self.pca_df = pd.DataFrame(
            self.X_pca,
            columns=[f"PC{i + 1}" for i in range(self.n_components)]
        )

        self.pca_df["label_id"] = y
        self.pca_df["label_name"] = self.pca_df["label_id"].apply(
            lambda label_id: target_names[label_id]
        )

        return self.pca_df

    def show_explained_variance(self):
        """
        打印每个主成分的解释方差比例。

        :return: 无返回值
        """
        print("\n========== PCA 解释方差比例 ==========")

        for index, ratio in enumerate(self.pca.explained_variance_ratio_):
            print(f"PC{index + 1} 解释方差比例：{ratio:.4f}")

        total_ratio = self.pca.explained_variance_ratio_.sum()
        print(f"前 {self.n_components} 个主成分累计解释方差比例：{total_ratio:.4f}")

    def plot_2d_pca(self):
        """
        绘制二维 PCA 散点图。

        :return: 无返回值
        """
        if self.n_components != 2:
            raise ValueError("只有 n_components=2 时才能绘制二维 PCA 图。")

        plt.figure(figsize=(8, 6))

        for label_name in self.pca_df["label_name"].unique():
            subset_df = self.pca_df[self.pca_df["label_name"] == label_name]

            plt.scatter(
                subset_df["PC1"],
                subset_df["PC2"],
                label=label_name,
                alpha=0.7
            )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA 2D Visualization")
        plt.legend()
        plt.grid(True)
        plt.show()


class PCAClassificationExperiment:
    """
    PCA 分类实验类。

    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, random_state: int = 42):
        """
        初始化 PCA 分类实验类。

        :param random_state: 随机种子
        :return: 无返回值
        """
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.pca = None

        self.raw_model = LogisticRegression(max_iter=1000)
        self.pca_model = LogisticRegression(max_iter=1000)

        self.X_train_scaled = None
        self.X_test_scaled = None

        self.X_train_pca = None
        self.X_test_pca = None

        self.raw_pred = None
        self.pca_pred = None

    def scale_data(self, X_train: np.ndarray, X_test: np.ndarray):
        """
        对训练集和测试集进行标准化。

        注意：
        训练集 fit_transform；
        测试集 transform。

        :param X_train: 训练集特征
        :param X_test: 测试集特征
        :return: 无返回值
        """
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

    def train_raw_feature_model(self, y_train: np.ndarray):
        """
        使用原始标准化特征训练逻辑回归模型。

        :param y_train: 训练集标签
        :return: 无返回值
        """
        self.raw_model.fit(self.X_train_scaled, y_train)
        print("\n原始 30 维特征模型训练完成。")

    def train_pca_feature_model(self, y_train: np.ndarray, n_components: int = 2):
        """
        使用 PCA 降维后的特征训练逻辑回归模型。

        :param y_train: 训练集标签
        :param n_components: PCA 主成分数量
        :return: 无返回值
        """
        self.pca = PCA(n_components=n_components)

        self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
        self.X_test_pca = self.pca.transform(self.X_test_scaled)

        self.pca_model.fit(self.X_train_pca, y_train)

        print(f"\nPCA {n_components} 维特征模型训练完成。")
        print("PCA 后训练集形状：", self.X_train_pca.shape)
        print("PCA 后测试集形状：", self.X_test_pca.shape)

        total_ratio = self.pca.explained_variance_ratio_.sum()
        print(f"PCA {n_components} 维累计解释方差比例：{total_ratio:.4f}")

    def evaluate_models(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray,
        target_names: np.ndarray
    ):
        """
        对比原始特征模型和 PCA 特征模型。

        :param y_train: 训练集标签
        :param y_test: 测试集标签
        :param target_names: 类别名称
        :return: 无返回值
        """
        self.raw_pred = self.raw_model.predict(self.X_test_scaled)
        self.pca_pred = self.pca_model.predict(self.X_test_pca)

        raw_accuracy = accuracy_score(y_test, self.raw_pred)
        pca_accuracy = accuracy_score(y_test, self.pca_pred)

        print("\n========== 原始特征模型 vs PCA 特征模型 ==========")
        print(f"原始 30 维特征测试集 Accuracy：{raw_accuracy:.4f}")
        print(f"PCA 降维特征测试集 Accuracy：{pca_accuracy:.4f}")

        print("\n========== 原始特征模型分类报告 ==========")
        print(classification_report(
            y_test,
            self.raw_pred,
            target_names=target_names
        ))

        print("\n========== PCA 特征模型分类报告 ==========")
        print(classification_report(
            y_test,
            self.pca_pred,
            target_names=target_names
        ))

        print("\n========== PCA 特征模型混淆矩阵 ==========")
        print(confusion_matrix(y_test, self.pca_pred))


class PCAComponentExperiment:
    """
    PCA 主成分数量实验类。

    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, random_state: int = 42):
        """
        初始化 PCA 主成分数量实验类。

        :param random_state: 随机种子
        :return: 无返回值
        """
        self.random_state = random_state
        self.result_df = None

    def compare_n_components(
        self,
        X_train_scaled: np.ndarray,
        X_test_scaled: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        比较不同 PCA 主成分数量下的模型效果。

        :param X_train_scaled: 标准化后的训练集特征
        :param X_test_scaled: 标准化后的测试集特征
        :param y_train: 训练集标签
        :param y_test: 测试集标签
        :return: 实验结果 DataFrame
        """
        component_values = [2, 3, 5, 8, 10, 15, 20, 30]

        results = []

        print("\n========== 不同 PCA 主成分数量对比 ==========")

        for n_components in component_values:
            pca = PCA(n_components=n_components)

            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_pca, y_train)

            y_pred = model.predict(X_test_pca)

            accuracy = accuracy_score(y_test, y_pred)
            explained_ratio = pca.explained_variance_ratio_.sum()

            results.append({
                "n_components": n_components,
                "explained_variance_ratio": explained_ratio,
                "test_accuracy": accuracy
            })

            print(
                f"n_components={n_components}, "
                f"累计解释方差比例={explained_ratio:.4f}, "
                f"测试集Accuracy={accuracy:.4f}"
            )

        self.result_df = pd.DataFrame(results)
        return self.result_df

    def plot_component_result(self):
        """
        绘制主成分数量与解释方差、准确率关系图。

        :return: 无返回值
        """
        if self.result_df is None:
            raise ValueError("请先运行 compare_n_components()。")

        plt.figure(figsize=(9, 5))

        plt.plot(
            self.result_df["n_components"],
            self.result_df["explained_variance_ratio"],
            marker="o",
            label="Explained Variance Ratio"
        )

        plt.plot(
            self.result_df["n_components"],
            self.result_df["test_accuracy"],
            marker="o",
            label="Test Accuracy"
        )

        plt.xlabel("n_components")
        plt.ylabel("Score")
        plt.title("PCA n_components vs Explained Variance and Accuracy")
        plt.xticks(self.result_df["n_components"])
        plt.legend()
        plt.grid(True)
        plt.show()


class PCAWorkflow:
    """
    PCA 完整学习流程类。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化 PCA 完整学习流程类。

        :param test_size: 测试集比例
        :param random_state: 随机种子
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state

        self.data_loader = BreastCancerDataLoader(
            test_size=self.test_size,
            random_state=self.random_state
        )

    def run(self):
        """
        运行完整 PCA 学习流程。

        :return: 无返回值
        """
        print("========== 第 16 天：PCA 降维 ==========")

        self.data_loader.load_data()
        self.data_loader.build_dataframe()
        self.data_loader.show_data_info()
        self.data_loader.split_data()

        visualizer = PCAVisualizer(n_components=2)
        visualizer.fit_transform(
            X=self.data_loader.X,
            y=self.data_loader.y,
            target_names=self.data_loader.target_names
        )
        visualizer.show_explained_variance()
        visualizer.plot_2d_pca()

        classification_experiment = PCAClassificationExperiment(
            random_state=self.random_state
        )

        classification_experiment.scale_data(
            X_train=self.data_loader.X_train,
            X_test=self.data_loader.X_test
        )

        classification_experiment.train_raw_feature_model(
            y_train=self.data_loader.y_train
        )

        classification_experiment.train_pca_feature_model(
            y_train=self.data_loader.y_train,
            n_components=2
        )

        classification_experiment.evaluate_models(
            y_train=self.data_loader.y_train,
            y_test=self.data_loader.y_test,
            target_names=self.data_loader.target_names
        )

        component_experiment = PCAComponentExperiment(
            random_state=self.random_state
        )

        result_df = component_experiment.compare_n_components(
            X_train_scaled=classification_experiment.X_train_scaled,
            X_test_scaled=classification_experiment.X_test_scaled,
            y_train=self.data_loader.y_train,
            y_test=self.data_loader.y_test
        )

        print("\n========== PCA 主成分数量实验结果 ==========")
        print(result_df)

        component_experiment.plot_component_result()


def main():
    """
    主函数。

    :return: 无返回值
    """
    workflow = PCAWorkflow(
        test_size=0.2,
        random_state=42
    )

    workflow.run()


if __name__ == "__main__":
    main()
十二、代码重点解释
1. 标准化
self.X_train_scaled = self.scaler.fit_transform(X_train)
self.X_test_scaled = self.scaler.transform(X_test)

PCA 前要标准化。

原因是：

PCA 关注方差；
不同特征尺度不同，会影响方差大小；
标准化后，各特征尺度更公平。
2. 创建 PCA
self.pca = PCA(n_components=2)

含义是：

把原来的 30 个特征降到 2 个主成分。

也就是：

30 维 → 2 维
3. 训练集 fit_transform，测试集 transform
self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
self.X_test_pca = self.pca.transform(self.X_test_scaled)

这和标准化规则一样。

训练集：

fit_transform：学习 PCA 方向，并转换训练集。

测试集：

transform：使用训练集学到的 PCA 方向转换测试集。

不能对测试集重新 fit PCA。

4. 查看解释方差比例
self.pca.explained_variance_ratio_

这会输出每个主成分保留的信息比例。

例如：

[0.44, 0.19]

表示：

PC1 保留 44% 信息；
PC2 保留 19% 信息；
前两个主成分合计保留 63% 信息。
5. PCA 可视化
plt.scatter(
    subset_df["PC1"],
    subset_df["PC2"],
    label=label_name,
    alpha=0.7
)

这一步把 30 维数据压缩成二维后画图。

你可以观察：

malignant 和 benign 是否大致分开；
是否有重叠区域；
PCA 二维图是否能帮助理解数据结构。
十三、运行后重点观察什么？
1. 原始数据维度

你会看到：

特征数据形状：(569, 30)

表示：

569 条样本；
每条样本 30 个特征。
2. PCA 二维后的形状

你会看到：

PCA 后训练集形状：(455, 2)
PCA 后测试集形状：(114, 2)

这表示：

原来的 30 个特征已经变成 2 个主成分。
3. 解释方差比例

你会看到类似：

PC1 解释方差比例：0.44
PC2 解释方差比例：0.19
前 2 个主成分累计解释方差比例：0.63

这说明二维 PCA 并没有保留全部信息，只保留了主要部分。

4. 原始特征模型 vs PCA 特征模型

你可能看到：

原始 30 维特征测试集 Accuracy：较高
PCA 2 维特征测试集 Accuracy：略低

这是正常的。

因为：

PCA 2 维只保留部分信息；
原始 30 维包含更多完整信息。

但是 PCA 2 维的好处是：

可以可视化；
模型更简单；
训练更快；
能压缩数据。
5. 不同 n_components 的结果

代码会比较：

2, 3, 5, 8, 10, 15, 20, 30

你重点看：

主成分越多，解释方差比例是否越高；
准确率是否一直提升；
多少个主成分已经能达到不错效果。
十四、PCA 主成分数量怎么选？

常见方法有 3 种。

1. 根据解释方差比例选择

例如保留：

90% 信息
95% 信息
99% 信息

代码可以这样写：

pca = PCA(n_components=0.95)

含义是：

自动选择能保留 95% 方差信息所需的主成分数量。
2. 根据模型效果选择

尝试不同主成分数量：

2, 5, 10, 20, 30

观察测试集准确率或交叉验证结果。

如果：

10 个主成分和 30 个原始特征效果差不多

那可以考虑用 10 个主成分。

3. 根据可视化需求选择

如果目标是画图：

二维图：n_components=2
三维图：n_components=3

二维 PCA 常用于观察数据分布和类别分离情况。

十五、PCA 的优缺点
1. 优点
可以降维；
可以压缩数据；
可以帮助可视化；
可以减少冗余；
可以一定程度降噪；
对高维数据很有用。
2. 缺点
主成分不如原始特征好解释；
PCA 是线性降维方法；
可能丢失部分有用信息；
对标准化敏感；
不一定提升模型效果。

尤其要记住：

PCA 不一定让准确率更高。

PCA 更常见的价值是：

压缩；
可视化；
降噪；
减少冗余。
十六、PCA 和模型训练的关系

PCA 可以放在模型训练流程中：

原始数据
    ↓
标准化
    ↓
PCA 降维
    ↓
训练模型
    ↓
预测新数据

使用 sklearn Pipeline 可以写成：

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

这个写法更适合真实项目，因为它能保证：

训练集 fit；
测试集 transform；
避免数据泄露；
流程更清晰。
十七、PCA 适合什么场景？

PCA 适合：

特征数量很多；
特征之间有相关性；
需要二维或三维可视化；
想压缩数据；
想减少噪声；
想提高训练速度。

例如：

图像特征压缩；
文本向量降维；
传感器数据降维；
金融指标降维；
医学特征可视化；
聚类前降维。
十八、PCA 不适合什么场景？

PCA 不太适合：

特征本身很少；
每个特征都需要强解释性；
数据关系明显是非线性的；
降维后模型效果明显下降；
业务要求必须知道每个原始字段的影响。

如果你需要解释：

到底是哪个原始字段影响最大？

PCA 可能不如随机森林特征重要性、逻辑回归系数、SHAP 等方法直观。

十九、今日练习
练习 1：解释 PCA

请用自己的话解释：

PCA 是什么？

参考方向：

PCA 是一种降维方法，它把原始多个特征重新组合成少数几个主成分，并尽量保留原始数据中的主要变化信息。
练习 2：修改 PCA 维度

把代码中的：

n_components=2

分别改成：

n_components=3
n_components=5
n_components=10
n_components=20

观察：

累计解释方差比例是否增加？
测试集准确率是否变化？
练习 3：使用 95% 方差信息

把：

PCA(n_components=2)

改成：

PCA(n_components=0.95)

观察：

PCA 自动保留了多少个主成分？
测试集准确率如何？

可以打印：

print(self.pca.n_components_)
练习 4：不标准化试试

把标准化去掉，直接 PCA：

self.X_train_pca = self.pca.fit_transform(X_train)
self.X_test_pca = self.pca.transform(X_test)

观察结果是否变化。

思考：

为什么 PCA 对特征尺度敏感？
练习 5：对 Iris 数据集做 PCA

把数据集换成：

from sklearn.datasets import load_iris

然后使用：

self.dataset = load_iris()

观察：

Iris 原始特征有几个？
PCA 降到 2 维后，三类花是否能分开？
前两个主成分解释方差比例是多少？
二十、今日学习检查表
检查项	是否掌握
能理解什么是降维	
能理解 PCA 的基本思想	
能理解主成分 PC1、PC2	
能理解解释方差比例	
能理解 PCA 为什么要标准化	
能区分 PCA 和特征选择	
能用 PCA 做二维可视化	
能用 PCA 降维后训练模型	
能比较不同 n_components 的效果	
能理解 PCA 的优缺点	
二十一、今日总结

第 16 天最重要的是这句话：

PCA 是把原始高维特征重新组合成少数几个主成分，并尽量保留数据中主要变化信息的降维方法。

你今天要掌握的核心流程是：

原始高维数据
    ↓
标准化
    ↓
PCA 学习主成分方向
    ↓
转换成低维特征
    ↓
可视化或建模

你现在应该能理解：

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

背后的含义是：

先让不同特征尺度一致；
再从训练集中学习 PCA 主成分方向；
把训练集和测试集都转换到新的低维空间。