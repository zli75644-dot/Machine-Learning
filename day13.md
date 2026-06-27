第 13 天的重点是：不要只凭一次训练/测试集划分来判断模型好坏，而是用交叉验证和调参，让模型选择更可靠。

已思考 8s
第 13 天：模型选择与交叉验证

前 12 天你已经学习了很多模型和数据处理方法：

线性回归
逻辑回归
KNN
决策树
随机森林
GBDT
数据预处理
特征工程

今天开始解决一个真实项目中非常关键的问题：

我有这么多模型，到底该选哪个？

如果只看一次训练集和测试集划分，结果可能不稳定。

所以第 13 天重点学习：

交叉验证 Cross Validation
模型对比
超参数调优
GridSearchCV
RandomizedSearchCV
一、今日学习目标

今天需要掌握 8 个内容：

理解训练集、验证集、测试集的区别；
理解为什么不能只看一次 train_test_split；
理解 K 折交叉验证；
使用 cross_val_score 比较多个模型；
使用 cross_validate 输出多个评估指标；
使用 GridSearchCV 做网格搜索调参；
使用 RandomizedSearchCV 做随机搜索调参；
输出一个模型排行榜，选择更可靠的模型。
二、为什么需要模型选择？

假设你现在有 5 个模型：

LogisticRegression
KNN
DecisionTree
RandomForest
GradientBoosting

你分别训练后得到测试集准确率：

模型	测试集 Accuracy
LogisticRegression	0.95
KNN	0.93
DecisionTree	0.90
RandomForest	0.96
GradientBoosting	0.97

你可能会说：

GradientBoosting 最高，所以选它。

但这里有一个问题：

这个结果可能只是当前这一次数据划分下的结果。

如果换一次训练集和测试集划分，结果可能变成：

模型	测试集 Accuracy
LogisticRegression	0.96
KNN	0.94
DecisionTree	0.91
RandomForest	0.97
GradientBoosting	0.95

这时最好的又变成了 RandomForest。

所以，模型选择不能只依赖一次划分结果。

三、训练集、验证集、测试集

在机器学习中，数据通常可以分成三部分：

训练集：模型学习参数用
验证集：模型选择和调参用
测试集：最终检验模型泛化能力用
1. 训练集 Train Set

训练集用于模型学习。

例如：

model.fit(X_train, y_train)

模型通过训练集学习：

权重
偏置
树的分裂规则
邻居样本
特征重要性
2. 验证集 Validation Set

验证集用于选择模型和调整超参数。

例如：

KNN 的 K 取 3 还是 5？
决策树 max_depth 取 3 还是 5？
随机森林 n_estimators 取 100 还是 300？
GBDT learning_rate 取 0.1 还是 0.05？

这些都应该通过验证集来比较。

3. 测试集 Test Set

测试集只能在最终阶段使用。

它的作用是：

模拟模型上线后遇到的新数据。

测试集不能反复参与调参。

如果你一直根据测试集结果调参数，测试集就不再“干净”，最后评估会偏乐观。

四、为什么需要交叉验证？

如果数据量不大，单独切出验证集会浪费数据。

例如你只有 150 条 Iris 数据。

如果划分：

训练集：90
验证集：30
测试集：30

训练集就更少了。

这时候可以用 K 折交叉验证。

五、什么是 K 折交叉验证？

K 折交叉验证就是把训练数据分成 K 份。

例如：

K = 5

把数据分成 5 份：

第 1 份
第 2 份
第 3 份
第 4 份
第 5 份

然后训练 5 次：

第 1 次：第 1 份做验证集，其余 4 份做训练集
第 2 次：第 2 份做验证集，其余 4 份做训练集
第 3 次：第 3 份做验证集，其余 4 份做训练集
第 4 次：第 4 份做验证集，其余 4 份做训练集
第 5 次：第 5 份做验证集，其余 4 份做训练集

最后得到 5 个分数：

0.95
0.96
0.93
0.97
0.94

然后求平均：

平均分 = 0.95

这个平均分比单次划分更稳定。

六、交叉验证的好处

交叉验证的好处是：

模型评估更稳定；
充分利用训练数据；
减少一次随机划分带来的偶然性；
更适合模型选择和超参数调优。

你可以这样理解：

一次考试可能有运气成分；
多次考试的平均成绩更能代表真实水平。
七、今日案例：乳腺癌分类模型选择

今天使用 sklearn 自带的乳腺癌数据集。

任务是：

根据细胞特征，判断肿瘤是恶性还是良性。

这是一个二分类任务。

我们会对比 5 个模型：

LogisticRegression
KNN
DecisionTree
RandomForest
GradientBoosting

并完成：

一次 train_test_split 对比
5 折交叉验证对比
多个指标评估
GridSearchCV 调参
RandomizedSearchCV 调参
最终测试集评估
八、今日完整代码

新建文件：

day13_model_selection_cross_validation.py

代码如下：

"""
第 13 天：模型选择与交叉验证。

本案例使用 sklearn 自带的乳腺癌数据集，演示真实机器学习项目中如何选择模型：

1. 加载数据
2. 划分训练集和测试集
3. 构建多个候选模型
4. 使用交叉验证比较模型
5. 输出模型排行榜
6. 使用 GridSearchCV 对随机森林调参
7. 使用 RandomizedSearchCV 对 GBDT 调参
8. 在最终测试集上评估最佳模型

重点：
测试集只用于最终评估，不应该反复参与调参。
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


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

        :param test_size: 测试集比例，例如 0.2 表示 20% 数据作为最终测试集
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

    def split_train_test(self):
        """
        划分训练集和最终测试集。

        注意：
        这里的测试集是最终测试集。
        后续模型选择和调参只在训练集内部做交叉验证。

        :return: 无返回值
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )

        print("\n========== 训练集 / 测试集划分 ==========")
        print("训练集特征形状：", self.X_train.shape)
        print("测试集特征形状：", self.X_test.shape)
        print("训练集标签形状：", self.y_train.shape)
        print("测试集标签形状：", self.y_test.shape)


class CandidateModelFactory:
    """
    候选模型工厂类。

    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, random_state: int = 42):
        """
        初始化候选模型工厂类。

        :param random_state: 随机种子
        :return: 无返回值
        """
        self.random_state = random_state

    def build_models(self) -> dict:
        """
        构建多个候选模型。

        说明：
        1. LogisticRegression、KNN 对特征尺度敏感，所以放入 StandardScaler
        2. 决策树、随机森林、GBDT 对特征尺度不敏感，一般不强制标准化

        :return: 候选模型字典
        """
        models = {
            "LogisticRegression": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000))
                ]
            ),
            "KNN": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier(n_neighbors=5))
                ]
            ),
            "DecisionTree": DecisionTreeClassifier(
                max_depth=5,
                random_state=self.random_state
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=120,
                max_depth=5,
                random_state=self.random_state
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=3,
                random_state=self.random_state
            )
        }

        return models


class CrossValidationModelSelector:
    """
    交叉验证模型选择类。

    :param models: 候选模型字典
    :param cv_splits: 交叉验证折数
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, models: dict, cv_splits: int = 5, random_state: int = 42):
        """
        初始化交叉验证模型选择类。

        :param models: 候选模型字典
        :param cv_splits: 交叉验证折数
        :param random_state: 随机种子
        :return: 无返回值
        """
        self.models = models
        self.cv_splits = cv_splits
        self.random_state = random_state

        self.cv = StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state
        )

        self.score_df = None
        self.multi_metric_df = None

    def compare_by_accuracy(self, X_train: np.ndarray, y_train: np.ndarray) -> pd.DataFrame:
        """
        使用 cross_val_score 按 accuracy 比较多个模型。

        :param X_train: 训练集特征
        :param y_train: 训练集标签
        :return: accuracy 对比结果 DataFrame
        """
        results = []

        print("\n========== 5 折交叉验证：Accuracy 对比 ==========")

        for model_name, model in self.models.items():
            scores = cross_val_score(
                estimator=model,
                X=X_train,
                y=y_train,
                cv=self.cv,
                scoring="accuracy"
            )

            result = {
                "model": model_name,
                "cv_accuracy_mean": scores.mean(),
                "cv_accuracy_std": scores.std(),
                "fold_scores": scores
            }

            results.append(result)

            print(
                f"{model_name}: "
                f"平均 Accuracy={scores.mean():.4f}, "
                f"标准差={scores.std():.4f}, "
                f"每折分数={np.round(scores, 4)}"
            )

        self.score_df = pd.DataFrame(results)
        self.score_df = self.score_df.sort_values(
            by="cv_accuracy_mean",
            ascending=False
        )

        return self.score_df

    def compare_by_multiple_metrics(self, X_train: np.ndarray, y_train: np.ndarray) -> pd.DataFrame:
        """
        使用 cross_validate 按多个指标比较模型。

        指标包括：
        1. accuracy
        2. precision
        3. recall
        4. f1

        :param X_train: 训练集特征
        :param y_train: 训练集标签
        :return: 多指标对比结果 DataFrame
        """
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1"
        }

        results = []

        print("\n========== 5 折交叉验证：多指标对比 ==========")

        for model_name, model in self.models.items():
            cv_result = cross_validate(
                estimator=model,
                X=X_train,
                y=y_train,
                cv=self.cv,
                scoring=scoring,
                return_train_score=True
            )

            result = {
                "model": model_name,
                "test_accuracy_mean": cv_result["test_accuracy"].mean(),
                "test_precision_mean": cv_result["test_precision"].mean(),
                "test_recall_mean": cv_result["test_recall"].mean(),
                "test_f1_mean": cv_result["test_f1"].mean(),
                "train_accuracy_mean": cv_result["train_accuracy"].mean(),
                "fit_time_mean": cv_result["fit_time"].mean()
            }

            results.append(result)

            print(
                f"{model_name}: "
                f"accuracy={result['test_accuracy_mean']:.4f}, "
                f"precision={result['test_precision_mean']:.4f}, "
                f"recall={result['test_recall_mean']:.4f}, "
                f"f1={result['test_f1_mean']:.4f}"
            )

        self.multi_metric_df = pd.DataFrame(results)
        self.multi_metric_df = self.multi_metric_df.sort_values(
            by="test_f1_mean",
            ascending=False
        )

        return self.multi_metric_df

    def show_model_ranking(self):
        """
        打印模型排行榜。

        :return: 无返回值
        """
        print("\n========== 模型排行榜：按 Accuracy ==========")
        print(
            self.score_df[
                ["model", "cv_accuracy_mean", "cv_accuracy_std"]
            ]
        )

        print("\n========== 模型排行榜：按 F1-score ==========")
        print(
            self.multi_metric_df[
                [
                    "model",
                    "test_accuracy_mean",
                    "test_precision_mean",
                    "test_recall_mean",
                    "test_f1_mean",
                    "fit_time_mean"
                ]
            ]
        )


class HyperParameterTuner:
    """
    超参数调优类。

    :param cv_splits: 交叉验证折数
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, cv_splits: int = 5, random_state: int = 42):
        """
        初始化超参数调优类。

        :param cv_splits: 交叉验证折数
        :param random_state: 随机种子
        :return: 无返回值
        """
        self.cv_splits = cv_splits
        self.random_state = random_state

        self.cv = StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state
        )

        self.grid_search = None
        self.random_search = None

    def tune_random_forest_by_grid_search(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        使用 GridSearchCV 对随机森林进行网格搜索调参。

        GridSearchCV 会穷举所有参数组合。

        :param X_train: 训练集特征
        :param y_train: 训练集标签
        :return: 训练完成的 GridSearchCV 对象
        """
        model = RandomForestClassifier(
            random_state=self.random_state
        )

        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 8, None],
            "min_samples_leaf": [1, 2, 4]
        }

        self.grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1",
            cv=self.cv,
            n_jobs=-1,
            verbose=1
        )

        print("\n========== GridSearchCV：随机森林调参 ==========")
        self.grid_search.fit(X_train, y_train)

        print("最佳参数：", self.grid_search.best_params_)
        print(f"最佳交叉验证 F1：{self.grid_search.best_score_:.4f}")

        return self.grid_search

    def tune_gbdt_by_random_search(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        使用 RandomizedSearchCV 对 GBDT 进行随机搜索调参。

        RandomizedSearchCV 不会穷举所有组合，而是随机抽取指定次数的组合。
        当参数空间较大时，RandomizedSearchCV 通常更省时间。

        :param X_train: 训练集特征
        :param y_train: 训练集标签
        :return: 训练完成的 RandomizedSearchCV 对象
        """
        model = GradientBoostingClassifier(
            random_state=self.random_state
        )

        param_distributions = {
            "n_estimators": [50, 100, 150, 200, 300],
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.2],
            "max_depth": [2, 3, 4, 5],
            "min_samples_leaf": [1, 2, 4, 6]
        }

        self.random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=20,
            scoring="f1",
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )

        print("\n========== RandomizedSearchCV：GBDT 调参 ==========")
        self.random_search.fit(X_train, y_train)

        print("最佳参数：", self.random_search.best_params_)
        print(f"最佳交叉验证 F1：{self.random_search.best_score_:.4f}")

        return self.random_search


class FinalModelEvaluator:
    """
    最终模型评估类。

    :param target_names: 目标类别名称
    :return: 无返回值
    """

    def __init__(self, target_names):
        """
        初始化最终模型评估类。

        :param target_names: 目标类别名称
        :return: 无返回值
        """
        self.target_names = target_names

    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "BestModel"):
        """
        在最终测试集上评估模型。

        注意：
        测试集只在模型选择和调参结束后使用。

        :param model: 已训练好的模型
        :param X_test: 测试集特征
        :param y_test: 测试集标签
        :param model_name: 模型名称
        :return: 无返回值
        """
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\n========== 最终测试集评估：{model_name} ==========")
        print(f"Accuracy：{accuracy:.4f}")
        print(f"Precision：{precision:.4f}")
        print(f"Recall：{recall:.4f}")
        print(f"F1-score：{f1:.4f}")

        print("\n混淆矩阵：")
        print(confusion_matrix(y_test, y_pred))

        print("\n分类报告：")
        print(classification_report(
            y_test,
            y_pred,
            target_names=self.target_names
        ))


class ModelSelectionWorkflow:
    """
    模型选择完整流程类。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化模型选择完整流程。

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

        self.models = None
        self.selector = None
        self.tuner = None
        self.evaluator = None

    def run(self):
        """
        运行完整模型选择流程。

        :return: 无返回值
        """
        print("========== 第 13 天：模型选择与交叉验证 ==========")

        self.data_loader.load_data()
        self.data_loader.build_dataframe()
        self.data_loader.show_data_info()
        self.data_loader.split_train_test()

        factory = CandidateModelFactory(
            random_state=self.random_state
        )
        self.models = factory.build_models()

        self.selector = CrossValidationModelSelector(
            models=self.models,
            cv_splits=5,
            random_state=self.random_state
        )

        self.selector.compare_by_accuracy(
            X_train=self.data_loader.X_train,
            y_train=self.data_loader.y_train
        )

        self.selector.compare_by_multiple_metrics(
            X_train=self.data_loader.X_train,
            y_train=self.data_loader.y_train
        )

        self.selector.show_model_ranking()

        self.tuner = HyperParameterTuner(
            cv_splits=5,
            random_state=self.random_state
        )

        random_forest_search = self.tuner.tune_random_forest_by_grid_search(
            X_train=self.data_loader.X_train,
            y_train=self.data_loader.y_train
        )

        gbdt_search = self.tuner.tune_gbdt_by_random_search(
            X_train=self.data_loader.X_train,
            y_train=self.data_loader.y_train
        )

        self.evaluator = FinalModelEvaluator(
            target_names=self.data_loader.target_names
        )

        self.evaluator.evaluate(
            model=random_forest_search.best_estimator_,
            X_test=self.data_loader.X_test,
            y_test=self.data_loader.y_test,
            model_name="GridSearchCV 最佳随机森林"
        )

        self.evaluator.evaluate(
            model=gbdt_search.best_estimator_,
            X_test=self.data_loader.X_test,
            y_test=self.data_loader.y_test,
            model_name="RandomizedSearchCV 最佳 GBDT"
        )


def main():
    """
    主函数。

    :return: 无返回值
    """
    workflow = ModelSelectionWorkflow(
        test_size=0.2,
        random_state=42
    )

    workflow.run()


if __name__ == "__main__":
    main()
九、代码重点解释
1. 为什么先划分最终测试集？

代码中先做：

self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(...)

这里的测试集是最终测试集。

之后的模型选择、交叉验证、调参都只在：

X_train
y_train

上进行。

原因是：

测试集必须保持干净，只用于最后一次评估。
2. 为什么 LogisticRegression 和 KNN 使用 Pipeline？

代码中：

"LogisticRegression": Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

原因是：

逻辑回归对特征尺度敏感；
KNN 对距离敏感；
所以它们通常需要标准化。

用 Pipeline 的好处是：

交叉验证时，每一折都会只在当前训练折上 fit 标准化器；
避免验证折信息泄露到训练过程。

这点非常重要。

3. StratifiedKFold 是什么？
self.cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

StratifiedKFold 是分层 K 折交叉验证。

它会尽量保证每一折中的类别比例和整体数据相似。

二分类任务中经常用它。

例如原始数据中：

良性 60%
恶性 40%

每一折也尽量保持接近这个比例。

4. cross_val_score 是什么？
scores = cross_val_score(
    estimator=model,
    X=X_train,
    y=y_train,
    cv=self.cv,
    scoring="accuracy"
)

它会返回每一折的分数。

例如：

[0.9560, 0.9670, 0.9450, 0.9780, 0.9560]

然后我们通常看：

平均值 mean
标准差 std

平均值越高越好，标准差越小越稳定。

5. cross_validate 是什么？

cross_validate 比 cross_val_score 更强。

它可以一次计算多个指标：

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

然后输出：

accuracy
precision
recall
f1
fit_time
score_time

这样比只看准确率更全面。

十、GridSearchCV 是什么？

GridSearchCV 是网格搜索调参。

例如随机森林有 3 个参数：

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 8, None],
    "min_samples_leaf": [1, 2, 4]
}

它会尝试所有组合。

组合数量是：

3 × 4 × 3 = 36 组

如果使用 5 折交叉验证：

36 × 5 = 180 次训练

所以 GridSearchCV 比较全面，但参数空间大时会很耗时。

十一、RandomizedSearchCV 是什么？

RandomizedSearchCV 是随机搜索调参。

它不会尝试全部组合，而是随机抽取一部分组合。

例如：

n_iter=20

表示随机尝试 20 组参数。

如果参数空间很大，随机搜索更省时间。

对比：

方法	搜索方式	优点	缺点
GridSearchCV	穷举所有组合	全面	参数多时很慢
RandomizedSearchCV	随机抽取组合	更快	不一定找到最优组合

入门阶段你可以这样选：

参数组合少：GridSearchCV
参数组合多：RandomizedSearchCV
十二、什么是超参数？

超参数就是模型训练前由人设置的参数。

例如：

KNN 的 n_neighbors
决策树的 max_depth
随机森林的 n_estimators
GBDT 的 learning_rate
GBDT 的 max_depth

它们不是模型从数据中直接学出来的，而是我们提前设定或搜索出来的。

对比一下：

类型	例子	            谁决定
参数	逻辑回归的 w 和 b	模型训练学出来
超参数	KNN 的 K 值     	人设置或搜索
超参数	随机森林树的数量	人设置或搜索
超参数	GBDT 学习率     	人设置或搜索
十三、运行后重点观察什么？

运行代码后，你重点看 5 个地方。

1. 模型 Accuracy 排行榜

你会看到类似：

model                 cv_accuracy_mean    cv_accuracy_std
LogisticRegression    0.9758              0.0120
RandomForest          0.9692              0.0150
GradientBoosting      0.9648              0.0180
KNN                   0.9604              0.0200
DecisionTree          0.9340              0.0300

看两个值：

cv_accuracy_mean：平均分，越高越好
cv_accuracy_std：波动，越小越稳定
2. 模型 F1 排行榜

F1-score 综合考虑 Precision 和 Recall。

在二分类任务中，比单看 Accuracy 更稳。

尤其当类别不均衡时，更应该看：

Precision
Recall
F1-score
3. 训练集和验证集差距

cross_validate 中我们保留了：

return_train_score=True

所以可以看到训练集分数和交叉验证分数。

如果：

train_accuracy 很高
test_accuracy 明显低

说明可能过拟合。

4. GridSearchCV 最佳参数

随机森林调参会输出：

最佳参数：
{
    'max_depth': 5,
    'min_samples_leaf': 2,
    'n_estimators': 100
}

这说明在当前搜索空间中，这组参数交叉验证效果最好。

5. 最终测试集评估

最后会在测试集上评估：

GridSearchCV 最佳随机森林
RandomizedSearchCV 最佳 GBDT

这个结果才是最终泛化能力参考。

注意：

不要用测试集反复调参。
十四、模型选择的一般流程

真实项目中，推荐流程是：

第 1 步：准备数据
第 2 步：划分最终测试集
第 3 步：在训练集上做交叉验证
第 4 步：比较多个候选模型
第 5 步：选择几个表现较好的模型
第 6 步：对候选模型做超参数调优
第 7 步：用最佳参数重新训练模型
第 8 步：在最终测试集上评估
第 9 步：保存模型
第 10 步：部署或进一步分析

你可以把它记成：

测试集只用于最后验收，不能参与平时练习。
十五、模型选择不是只看最高分

模型选择时，不要只看一个分数最高。

还要综合考虑：

准确率
Precision
Recall
F1-score
训练时间
预测速度
模型稳定性
模型可解释性
是否容易部署
是否容易调参
业务能否接受

例如：

LogisticRegression 分数略低，但解释性强、速度快；
RandomForest 分数较高，稳定，但模型较大；
GBDT 分数高，但调参更复杂；
DecisionTree 分数低一些，但可解释性最强。

所以真实项目中不是永远选择分数最高的模型，而是选择：

综合效果最适合业务目标的模型。
十六、不同模型适用场景简单总结
模型	                    适合场景            	优点	            缺点
LogisticRegression	二分类基线、解释性要求高	快、简单、可解释	    非线性能力弱
KNN	                    小数据、边界直观	    简单直观	        预测慢、依赖标准化
DecisionTree	            规则解释	        可视化、好理解	        容易过拟合
RandomForest	        稳定表格建模	        稳定、调参简单  	解释性弱于单棵树
GradientBoosting	    追求较高效果	        拟合能力强	            调参更复杂
十七、今日练习
练习 1：解释交叉验证

请用自己的话解释：

为什么交叉验证比一次 train_test_split 更可靠？

参考方向：

一次划分可能有偶然性；
交叉验证会多次训练和验证；
平均分更稳定；
能更可靠地比较模型。
练习 2：修改交叉验证折数

把：

cv_splits=5

改成：

cv_splits=3
cv_splits=10

观察：

平均分是否变化？
标准差是否变化？
运行时间是否变化？
练习 3：修改 KNN 参数

在 CandidateModelFactory 中把：

KNeighborsClassifier(n_neighbors=5)

分别改成：

KNeighborsClassifier(n_neighbors=3)
KNeighborsClassifier(n_neighbors=7)
KNeighborsClassifier(n_neighbors=11)

观察交叉验证结果。

练习 4：修改随机森林搜索空间

把随机森林参数搜索改成：

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, 10],
    "min_samples_leaf": [1, 2, 3]
}

观察最佳参数是否变化。

练习 5：把评分指标改成 Recall

在医疗任务中，如果你更关注“尽量不要漏掉恶性病例”，可以把：

scoring="f1"

改成：

scoring="recall"

然后观察最佳参数是否变化。

注意：

在 sklearn 乳腺癌数据集中，默认正类是 label=1，也就是 benign。
如果你要特别关注 malignant，需要进一步设置 pos_label=0 的自定义 scorer。

这个点后面做真实项目时要特别小心。

十八、今日学习检查表
检查项	是否掌握
能理解训练集、验证集、测试集区别	
能理解测试集不能反复调参	
能理解 K 折交叉验证	
能使用 cross_val_score	
能使用 cross_validate	
能理解平均分和标准差	
能使用 GridSearchCV	
能使用 RandomizedSearchCV	
能理解超参数	
能输出模型排行榜并选择模型	
十九、今日总结

第 13 天最重要的是这句话：

模型选择不能只看一次测试结果，要用交叉验证比较多个模型，再用调参搜索选择更可靠的参数组合。

今天你应该掌握下面这条完整思路：

先划分最终测试集
    ↓
训练集内部做交叉验证
    ↓
比较多个模型
    ↓
选择候选模型
    ↓
GridSearchCV / RandomizedSearchCV 调参
    ↓
用最佳模型在最终测试集上评估

你现在应该能理解这些代码的意义：

cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

cross_validate(model, X_train, y_train, cv=5, scoring=scoring)

GridSearchCV(model, param_grid, cv=5, scoring="f1")

RandomizedSearchCV(model, param_distributions, n_iter=20, cv=5)