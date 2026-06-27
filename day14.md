第 14 天重点解决一个核心判断：训练集分数高不一定代表模型好，要看测试集表现是否同步好。今天会用决策树深度变化来直观看懂欠拟合、合适拟合和过拟合。

已思考 13s
第 14 天：过拟合与欠拟合

今天学习机器学习里非常核心的问题：

过拟合 Overfitting
欠拟合 Underfitting
泛化能力 Generalization

前面你已经学过很多模型：

线性回归
逻辑回归
KNN
决策树
随机森林
GBDT

但是模型训练出来之后，不是只看它在训练集上的效果，而是要看它面对新数据时表现如何。

今天的核心问题是：

为什么训练集准确率很高，测试集准确率却可能很低？
一、今日学习目标

今天需要掌握 8 个内容：

什么是欠拟合；
什么是过拟合；
什么是泛化能力；
如何通过训练集分数和测试集分数判断模型状态；
理解模型复杂度对结果的影响；
理解决策树 max_depth 和过拟合的关系；
理解正则化的基本思想；
用代码画出 max_depth 与训练集、测试集准确率变化曲线。
二、什么是泛化能力？

泛化能力指的是：

模型在没见过的新数据上的表现能力。

机器学习真正追求的不是：

训练集上预测得很准

而是：

新数据上预测得也准

例如学生考试：

训练集 = 平时做过的练习题
测试集 = 考试中的新题

如果一个学生只会背练习题答案，平时练习分数很高，但考试换个题就不会了，这就类似过拟合。

三、什么是欠拟合？

欠拟合就是模型太简单，连训练集都学不好。

表现是：

训练集分数低；
测试集分数也低。

例如：

训练集 Accuracy = 0.70
测试集 Accuracy = 0.68

这说明模型没有学到数据中的主要规律。

欠拟合的常见原因
模型太简单；
特征太少；
特征表达能力太弱；
训练次数太少；
正则化太强；
决策树深度太浅；
数据本身噪声太大。
欠拟合例子

假设真实规律是：

学生是否挂科 = 学习时长 + 出勤率 + 作业完成率 + 平时测验成绩共同决定

但你只给模型一个特征：

年龄

那模型很可能学不好。

因为年龄和挂科关系不强。

这就是特征不足导致欠拟合。

四、什么是过拟合？

过拟合就是模型太复杂，把训练集中的细节、噪声、偶然情况都记住了。

表现是：

训练集分数很高；
测试集分数明显低。

例如：

训练集 Accuracy = 1.00
测试集 Accuracy = 0.78

这说明模型在训练集上表现完美，但面对新数据时效果不好。

过拟合的常见原因
模型太复杂；
特征太多但样本太少；
决策树太深；
KNN 中 K 值太小；
训练轮数太多；
没有正则化；
数据噪声被模型学进去了。
过拟合例子

假设你训练一棵决策树。

如果不限制深度，它可能学出非常细的规则：

如果 花瓣长度 <= 2.45，判断为 setosa
否则如果 花瓣宽度 <= 1.75，判断为 versicolor
否则如果 花萼宽度 <= 2.95，判断为 virginica
否则如果 花萼长度 <= 6.35，判断为 versicolor
否则继续分裂……

它可能把训练集中每一个特殊样本都单独记住。

训练集准确率非常高，但新样本来了，它可能判断错误。

五、三种模型状态对比
状态	训练集表现	测试集表现	    问题
欠拟合	    低          低	    模型太简单，没学会
合适拟合	高	        高      泛化能力较好
过拟合	    很高	明显较低	模型太复杂，记住了噪声

可以这样记：

欠拟合：训练集都学不好。
合适拟合：训练集和测试集都不错。
过拟合：训练集很好，测试集不行。
六、模型复杂度和拟合状态

模型复杂度可以理解为模型表达能力。

例如决策树：

max_depth=1：模型很简单，容易欠拟合；
max_depth=3：复杂度适中，可能效果较好；
max_depth=None：树无限制生长，容易过拟合。

KNN 也有类似情况：

K 很大：模型太平滑，可能欠拟合；
K 很小：模型太敏感，可能过拟合。

逻辑回归中：

正则化太强：可能欠拟合；
正则化太弱：可能过拟合。
七、偏差和方差的直观理解

过拟合和欠拟合经常和两个概念有关：

Bias：偏差
Variance：方差
1. 偏差 Bias

偏差可以理解为：

模型本身离真实规律有多远。

如果模型太简单，就容易高偏差。

例如真实规律很复杂，但你用一条直线去拟合，模型很难学好。

这就是欠拟合。

2. 方差 Variance

方差可以理解为：

模型对训练数据变化有多敏感。

如果换一点训练数据，模型结果就变化很大，说明方差高。

复杂决策树通常方差较高，容易过拟合。

3. 偏差和方差对应关系
状态	偏差	方差
欠拟合	高	    低
合适拟合	适中	适中
过拟合	低	    高

通俗理解：

欠拟合：模型太笨。
过拟合：模型太聪明，但聪明过头，开始背答案。
八、如何缓解欠拟合？

如果模型欠拟合，可以考虑：

换更复杂的模型；
增加有效特征；
减少过强的正则化；
增加训练轮数；
提高决策树 max_depth；
做更好的特征工程；
使用非线性模型，如随机森林、GBDT。

例如：

逻辑回归欠拟合，可以尝试随机森林；
决策树 max_depth=1 欠拟合，可以改成 max_depth=3 或 5；
特征太少，可以增加业务特征。
九、如何缓解过拟合？

如果模型过拟合，可以考虑：

降低模型复杂度；
减少决策树深度；
增大 min_samples_leaf；
使用正则化；
增加训练数据；
删除无用特征或噪声特征；
使用交叉验证选择参数；
使用随机森林等集成模型；
早停 Early Stopping。

例如决策树过拟合：

DecisionTreeClassifier(max_depth=3)

比不限制深度更稳。

随机森林过拟合时可以调：

RandomForestClassifier(
    max_depth=5,
    min_samples_leaf=3,
    n_estimators=200
)
十、今日案例：观察决策树深度如何影响过拟合

今天使用 sklearn 自带的乳腺癌数据集。

任务是：

根据细胞特征，判断肿瘤是恶性还是良性。

我们会训练多棵不同深度的决策树：

max_depth = 1
max_depth = 2
max_depth = 3
max_depth = 4
max_depth = 5
max_depth = 8
max_depth = 12
max_depth = None

然后观察：

训练集 Accuracy
测试集 Accuracy
训练集和测试集差距
十一、今日完整代码

新建文件：

day14_overfitting_underfitting.py

代码如下：

"""
第 14 天：过拟合与欠拟合。

本案例使用 sklearn 自带的乳腺癌数据集，
通过观察不同 max_depth 下决策树的训练集和测试集准确率，
理解：

1. 什么是欠拟合
2. 什么是过拟合
3. 什么是合适拟合
4. 模型复杂度如何影响泛化能力
5. 如何通过训练集分数和测试集分数判断模型状态
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class BreastCancerDataManager:
    """
    乳腺癌数据管理类。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化乳腺癌数据管理类。

        :param test_size: 测试集比例，例如 0.2 表示 20% 数据作为测试集
        :param random_state: 随机种子，用于保证实验结果可复现
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

        print("\n========== 标签 ID 对应关系 ==========")
        for label_id, label_name in enumerate(self.target_names):
            print(f"{label_id} -> {label_name}")

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


class DecisionTreeDepthExperiment:
    """
    决策树深度实验类。

    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, random_state: int = 42):
        """
        初始化决策树深度实验类。

        :param random_state: 随机种子
        :return: 无返回值
        """
        self.random_state = random_state
        self.result_df = None

    def judge_fit_status(self, train_accuracy: float, test_accuracy: float) -> str:
        """
        根据训练集和测试集准确率粗略判断模型状态。

        说明：
        这里只是教学演示用的简单规则，真实项目还需要结合交叉验证、
        业务指标、数据量、类别分布等综合判断。

        :param train_accuracy: 训练集准确率
        :param test_accuracy: 测试集准确率
        :return: 模型状态描述
        """
        gap = train_accuracy - test_accuracy

        if train_accuracy < 0.85 and test_accuracy < 0.85:
            return "可能欠拟合"

        if train_accuracy >= 0.95 and gap >= 0.08:
            return "可能过拟合"

        return "相对合适"

    def run_depth_comparison(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        对比不同 max_depth 下决策树的训练集和测试集表现。

        :param X_train: 训练集特征
        :param X_test: 测试集特征
        :param y_train: 训练集标签
        :param y_test: 测试集标签
        :return: 实验结果 DataFrame
        """
        depth_values = [1, 2, 3, 4, 5, 8, 12, None]

        results = []

        print("\n========== 决策树 max_depth 对比实验 ==========")

        for depth in depth_values:
            model = DecisionTreeClassifier(
                max_depth=depth,
                random_state=self.random_state
            )

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            gap = train_accuracy - test_accuracy

            fit_status = self.judge_fit_status(
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy
            )

            results.append({
                "max_depth": str(depth),
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "gap": gap,
                "fit_status": fit_status
            })

            print(
                f"max_depth={depth}, "
                f"训练集准确率={train_accuracy:.4f}, "
                f"测试集准确率={test_accuracy:.4f}, "
                f"差距={gap:.4f}, "
                f"状态={fit_status}"
            )

        self.result_df = pd.DataFrame(results)
        return self.result_df

    def plot_depth_curve(self):
        """
        绘制 max_depth 与训练集、测试集准确率关系图。

        :return: 无返回值
        """
        if self.result_df is None:
            raise ValueError("请先运行 run_depth_comparison() 得到实验结果。")

        plt.figure(figsize=(10, 6))

        x_values = self.result_df["max_depth"]

        plt.plot(
            x_values,
            self.result_df["train_accuracy"],
            marker="o",
            label="Train Accuracy"
        )

        plt.plot(
            x_values,
            self.result_df["test_accuracy"],
            marker="o",
            label="Test Accuracy"
        )

        plt.xlabel("max_depth")
        plt.ylabel("Accuracy")
        plt.title("Decision Tree max_depth vs Accuracy")
        plt.ylim(0.75, 1.05)
        plt.legend()
        plt.grid(True)
        plt.show()


class CrossValidationComplexityExperiment:
    """
    交叉验证复杂度实验类。

    :param random_state: 随机种子
    :param cv_splits: 交叉验证折数
    :return: 无返回值
    """

    def __init__(self, random_state: int = 42, cv_splits: int = 5):
        """
        初始化交叉验证复杂度实验类。

        :param random_state: 随机种子
        :param cv_splits: 交叉验证折数
        :return: 无返回值
        """
        self.random_state = random_state
        self.cv_splits = cv_splits

        self.cv = StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state
        )

        self.result_df = None

    def run_cross_validation_depth_comparison(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> pd.DataFrame:
        """
        使用交叉验证对比不同 max_depth 的稳定性。

        :param X_train: 训练集特征
        :param y_train: 训练集标签
        :return: 交叉验证结果 DataFrame
        """
        depth_values = [1, 2, 3, 4, 5, 8, 12, None]

        results = []

        print("\n========== 交叉验证：不同 max_depth 对比 ==========")

        for depth in depth_values:
            model = DecisionTreeClassifier(
                max_depth=depth,
                random_state=self.random_state
            )

            scores = cross_val_score(
                estimator=model,
                X=X_train,
                y=y_train,
                cv=self.cv,
                scoring="accuracy"
            )

            results.append({
                "max_depth": str(depth),
                "cv_accuracy_mean": scores.mean(),
                "cv_accuracy_std": scores.std()
            })

            print(
                f"max_depth={depth}, "
                f"CV平均准确率={scores.mean():.4f}, "
                f"CV标准差={scores.std():.4f}, "
                f"每折分数={np.round(scores, 4)}"
            )

        self.result_df = pd.DataFrame(results)
        return self.result_df


class RegularizationExperiment:
    """
    逻辑回归正则化实验类。

    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, random_state: int = 42):
        """
        初始化逻辑回归正则化实验类。

        :param random_state: 随机种子
        :return: 无返回值
        """
        self.random_state = random_state
        self.result_df = None

    def run_c_comparison(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        对比不同 C 值下逻辑回归模型效果。

        在 sklearn 的 LogisticRegression 中：
        C 越小，正则化越强；
        C 越大，正则化越弱。

        :param X_train: 训练集特征
        :param X_test: 测试集特征
        :param y_train: 训练集标签
        :param y_test: 测试集标签
        :return: 实验结果 DataFrame
        """
        c_values = [0.001, 0.01, 0.1, 1, 10, 100]

        results = []

        print("\n========== 逻辑回归正则化 C 参数对比 ==========")

        for c_value in c_values:
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=c_value,
                            max_iter=1000,
                            random_state=self.random_state
                        )
                    )
                ]
            )

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            gap = train_accuracy - test_accuracy

            results.append({
                "C": c_value,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "gap": gap
            })

            print(
                f"C={c_value}, "
                f"训练集准确率={train_accuracy:.4f}, "
                f"测试集准确率={test_accuracy:.4f}, "
                f"差距={gap:.4f}"
            )

        self.result_df = pd.DataFrame(results)
        return self.result_df


class ModelComparisonForGeneralization:
    """
    不同模型泛化能力对比类。

    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, random_state: int = 42):
        """
        初始化不同模型泛化能力对比类。

        :param random_state: 随机种子
        :return: 无返回值
        """
        self.random_state = random_state
        self.result_df = None

    def build_models(self) -> dict:
        """
        构建用于对比的模型。

        :return: 模型字典
        """
        models = {
            "SimpleTree_depth1": DecisionTreeClassifier(
                max_depth=1,
                random_state=self.random_state
            ),
            "GoodTree_depth4": DecisionTreeClassifier(
                max_depth=4,
                random_state=self.random_state
            ),
            "ComplexTree_no_limit": DecisionTreeClassifier(
                max_depth=None,
                random_state=self.random_state
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_leaf=2,
                random_state=self.random_state
            )
        }

        return models

    def compare_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        对比不同复杂度模型的训练集和测试集表现。

        :param X_train: 训练集特征
        :param X_test: 测试集特征
        :param y_train: 训练集标签
        :param y_test: 测试集标签
        :return: 模型对比结果 DataFrame
        """
        models = self.build_models()

        results = []

        print("\n========== 不同模型泛化能力对比 ==========")

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            gap = train_accuracy - test_accuracy

            results.append({
                "model": model_name,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "gap": gap
            })

            print(
                f"{model_name}, "
                f"训练集准确率={train_accuracy:.4f}, "
                f"测试集准确率={test_accuracy:.4f}, "
                f"差距={gap:.4f}"
            )

        self.result_df = pd.DataFrame(results)
        return self.result_df


class OverfittingUnderfittingWorkflow:
    """
    过拟合与欠拟合完整实验流程类。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化完整实验流程。

        :param test_size: 测试集比例
        :param random_state: 随机种子
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state

        self.data_manager = BreastCancerDataManager(
            test_size=self.test_size,
            random_state=self.random_state
        )

    def run(self):
        """
        运行完整实验流程。

        :return: 无返回值
        """
        print("========== 第 14 天：过拟合与欠拟合 ==========")

        self.data_manager.load_data()
        self.data_manager.build_dataframe()
        self.data_manager.show_data_info()
        self.data_manager.split_data()

        depth_experiment = DecisionTreeDepthExperiment(
            random_state=self.random_state
        )

        depth_result_df = depth_experiment.run_depth_comparison(
            X_train=self.data_manager.X_train,
            X_test=self.data_manager.X_test,
            y_train=self.data_manager.y_train,
            y_test=self.data_manager.y_test
        )

        print("\n========== 决策树深度实验结果 ==========")
        print(depth_result_df)

        depth_experiment.plot_depth_curve()

        cv_experiment = CrossValidationComplexityExperiment(
            random_state=self.random_state,
            cv_splits=5
        )

        cv_result_df = cv_experiment.run_cross_validation_depth_comparison(
            X_train=self.data_manager.X_train,
            y_train=self.data_manager.y_train
        )

        print("\n========== 交叉验证复杂度实验结果 ==========")
        print(cv_result_df)

        regularization_experiment = RegularizationExperiment(
            random_state=self.random_state
        )

        regularization_result_df = regularization_experiment.run_c_comparison(
            X_train=self.data_manager.X_train,
            X_test=self.data_manager.X_test,
            y_train=self.data_manager.y_train,
            y_test=self.data_manager.y_test
        )

        print("\n========== 正则化实验结果 ==========")
        print(regularization_result_df)

        model_comparison = ModelComparisonForGeneralization(
            random_state=self.random_state
        )

        model_comparison_df = model_comparison.compare_models(
            X_train=self.data_manager.X_train,
            X_test=self.data_manager.X_test,
            y_train=self.data_manager.y_train,
            y_test=self.data_manager.y_test
        )

        print("\n========== 模型泛化能力对比结果 ==========")
        print(model_comparison_df)


def main():
    """
    主函数。

    :return: 无返回值
    """
    workflow = OverfittingUnderfittingWorkflow(
        test_size=0.2,
        random_state=42
    )

    workflow.run()


if __name__ == "__main__":
    main()
十二、代码重点解释
1. 为什么用决策树演示过拟合？

因为决策树的复杂度非常容易控制。

核心参数是：

max_depth

如果：

max_depth=1

树非常浅，模型很简单，容易欠拟合。

如果：

max_depth=None

树不限制深度，容易把训练集学得特别细，可能过拟合。

2. 如何判断模型状态？

代码中有这个方法：

def judge_fit_status(self, train_accuracy: float, test_accuracy: float) -> str:

它通过训练集和测试集准确率粗略判断：

训练集低，测试集也低：可能欠拟合；
训练集很高，测试集明显低：可能过拟合；
训练集和测试集都较高，差距不大：相对合适。

注意：

这是教学演示规则，不是绝对标准。

真实项目中，还要结合：

交叉验证；
样本量；
业务指标；
Precision；
Recall；
F1-score；
模型稳定性。
3. 为什么要看 gap？

代码里计算：

gap = train_accuracy - test_accuracy

这个差值很重要。

如果：

gap 很小

说明训练集和测试集表现接近。

如果：

gap 很大

说明模型可能在训练集上学得太细，对新数据泛化不好。

4. 为什么还要做交叉验证？

一次 train/test 划分可能有偶然性。

所以代码中增加了：

cross_val_score()

用于观察不同 max_depth 在 5 折交叉验证中的平均表现。

重点看：

cv_accuracy_mean：平均效果；
cv_accuracy_std：波动大小。

如果平均分高且标准差小，说明模型更稳定。

5. 正则化实验怎么看？

代码中逻辑回归使用参数：

C

在 sklearn 中：

C 越小，正则化越强；
C 越大，正则化越弱。

如果正则化太强，模型可能欠拟合。

如果正则化太弱，模型可能过拟合。

你重点观察：

不同 C 下训练集和测试集准确率是否变化；
训练集和测试集差距是否变化。
十三、运行后重点观察什么？

运行代码后，重点看 4 个部分。

1. max_depth=1 的结果

你可能看到：

训练集准确率较低；
测试集准确率也较低。

这通常说明树太浅，模型表达能力不足。

也就是：

欠拟合。
2. max_depth=3 或 4 的结果

你可能看到：

训练集准确率较高；
测试集准确率也较高；
二者差距不大。

这通常说明模型复杂度比较合适。

3. max_depth=None 的结果

你可能看到：

训练集准确率接近 1.0000；
测试集准确率没有同步提升；
训练集和测试集差距变大。

这可能说明模型记住了训练集中的细节。

也就是：

过拟合。
4. 随机森林对比

随机森林通常比单棵复杂树更稳定。

原因是：

单棵树容易受训练数据影响；
随机森林通过多棵树投票降低不稳定性。

所以你可以观察：

ComplexTree_no_limit 和 RandomForest 的 gap 谁更大？
十四、不同模型如何控制过拟合？
1. 决策树

常用参数：

DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=3,
    min_samples_split=6
)

控制思路：

降低 max_depth；
增大 min_samples_leaf；
增大 min_samples_split。
2. 随机森林

常用参数：

RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=3
)

控制思路：

限制每棵树深度；
增加叶子节点最小样本数；
增加树数量提高稳定性。
3. 逻辑回归

常用参数：

LogisticRegression(C=0.1)

控制思路：

减小 C，增强正则化；
增大 C，减弱正则化。
4. KNN

常用参数：

KNeighborsClassifier(n_neighbors=7)

控制思路：

K 太小容易过拟合；
K 太大容易欠拟合；
通过交叉验证选择合适 K。
5. GBDT

常用参数：

GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3
)

控制思路：

降低 learning_rate；
限制 max_depth；
控制 n_estimators；
增大 min_samples_leaf。
十五、训练集和测试集结果如何解读？

你可以用下面这张表判断：

训练集分数	测试集分数	状态
低	低	欠拟合
高	高	拟合较好
很高	明显较低	过拟合
中等	中等	可能还可以提升
高	略低	正常现象

注意：

训练集分数一般会高于测试集分数。

这是正常的。

真正需要警惕的是：

训练集明显高于测试集。
十六、一个重要误区：测试集分数低，不一定都是过拟合

测试集分数低可能有多种原因：

模型过拟合；
模型欠拟合；
特征质量差；
数据标签有问题；
训练集和测试集分布不一致；
样本数量太少；
类别不均衡；
评估指标选错。

所以判断问题时不能只看一个数字。

建议按下面顺序检查：

训练集分数怎么样？
测试集分数怎么样？
二者差距多大？
交叉验证平均分怎么样？
交叉验证标准差大不大？
类别分布是否均衡？
特征是否有泄露？
特征是否有效？
十七、真实项目中的处理流程

如果发现欠拟合：

增加特征；
换更强模型；
提高模型复杂度；
减少过强正则化；
增加训练轮数；
做更好的特征工程。

如果发现过拟合：

降低模型复杂度；
增加正则化；
增加数据量；
删除噪声特征；
使用交叉验证调参；
使用集成模型；
做数据增强；
使用早停。
十八、今日练习
练习 1：判断模型状态

请判断下面几组结果属于欠拟合、过拟合还是比较合适。

训练集 Accuracy	测试集 Accuracy	状态
        0.70	        0.68	欠拟合
        0.98	        0.96	比较合适
        1.00	        0.78	过拟合
        0.83	        0.82	可能欠拟合或模型能力一般
        0.95	        0.74	过拟合
练习 2：修改决策树深度

把代码中的：

depth_values = [1, 2, 3, 4, 5, 8, 12, None]

改成：

depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]

观察：

训练集准确率是否越来越高？
测试集准确率是否一直提高？
什么时候二者差距开始变大？
练习 3：调整 min_samples_leaf

在决策树模型中增加：

min_samples_leaf=3

例如：

model = DecisionTreeClassifier(
    max_depth=depth,
    min_samples_leaf=3,
    random_state=self.random_state
)

观察：

训练集准确率是否降低？
测试集准确率是否更稳定？
过拟合是否有所缓解？
练习 4：观察正则化 C

在逻辑回归实验中观察：

C = 0.001
C = 0.01
C = 0.1
C = 1
C = 10
C = 100

回答：

C 很小时，模型是否更简单？
C 很大时，训练集和测试集差距是否变化？
练习 5：比较单棵树和随机森林

观察：

ComplexTree_no_limit
RandomForest

回答：

哪个训练集准确率更高？
哪个测试集更稳定？
哪个 gap 更小？
为什么随机森林通常比单棵树更稳定？
十九、今日学习检查表
检查项	是否掌握
能理解泛化能力	
能理解欠拟合	
能理解过拟合	
能通过训练集和测试集分数判断模型状态	
能理解模型复杂度	
能理解 max_depth 和过拟合关系	
能理解 gap 的含义	
能理解正则化基本思想	
能知道如何缓解欠拟合	
能知道如何缓解过拟合	
二十、今日总结

第 14 天最重要的是这句话：

欠拟合是模型太简单，训练集和测试集都学不好；
过拟合是模型太复杂，训练集很好但测试集明显变差。

你现在应该能看懂：

model = DecisionTreeClassifier(max_depth=1)

可能欠拟合，因为树太浅。

也能理解：

model = DecisionTreeClassifier(max_depth=None)

可能过拟合，因为树可以无限制生长，把训练集细节记住。

今天要形成一个判断习惯：

不要只看训练集分数；
不要只看一次测试集分数；
要看训练集、测试集、交叉验证、指标差距和业务目标。