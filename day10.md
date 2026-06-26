第 10 天：GBDT、XGBoost、LightGBM 思想

前面第 8 天学习了 决策树，第 9 天学习了 随机森林。

今天进入表格数据建模中非常重要的一类模型：

Boosting 集成学习
GBDT
XGBoost
LightGBM

这些模型在很多结构化数据任务中非常常用，例如：

用户是否流失
订单是否异常
贷款是否违约
商品销量预测
广告点击率预测
学生成绩预测
电商复购预测
医疗风险预测

今天重点不是死记复杂公式，而是先理解它们的核心思想。

一、今日学习目标

今天需要掌握 8 个内容：

理解 Bagging 和 Boosting 的区别；
理解随机森林和 GBDT 的区别；
理解 GBDT 是如何一棵树接一棵树修正错误的；
理解 XGBoost 为什么比传统 GBDT 更强；
理解 LightGBM 为什么速度更快；
使用 sklearn 的 GradientBoostingClassifier 完成分类任务；
对比决策树、随机森林、GBDT 的效果；
输出并分析特征重要性。
二、先回顾随机森林

第 9 天我们学过随机森林。

随机森林的思想是：

训练很多棵决策树；
每棵树相对独立；
分类时大家投票；
票数最多的类别作为最终结果。

可以理解为：

多个老师分别打分，最后少数服从多数。

随机森林属于：

Bagging
三、什么是 Bagging？

Bagging 的核心思想是：

并行训练多个模型，然后把结果聚合起来。

随机森林就是典型 Bagging。

流程是：

原始训练集
    ↓
随机抽样生成多个子训练集
    ↓
每个子训练集训练一棵决策树
    ↓
多棵树并行预测
    ↓
分类任务投票，回归任务取平均

Bagging 的重点是：

降低模型方差，让模型更加稳定。

通俗理解：

一个人判断可能不稳定；
多个人一起判断更稳。
四、什么是 Boosting？

Boosting 的核心思想和 Bagging 不一样。

Boosting 是：

多个模型不是并行训练，而是一个接一个训练。

它的核心是：

后一个模型重点修正前一个模型犯的错误。

可以理解为：

第 1 个学生先做题；
第 2 个学生专门改第 1 个学生错的地方；
第 3 个学生继续改前面还没改好的地方；
不断迭代，整体答案越来越准确。

Boosting 的流程是：

训练第 1 个弱模型
    ↓
找出预测错误或预测不好的地方
    ↓
训练第 2 个弱模型去修正错误
    ↓
继续训练第 3 个弱模型
    ↓
不断累加多个弱模型
    ↓
得到一个强模型

Boosting 的重点是：

降低偏差，让模型一步步变强。
五、Bagging 和 Boosting 的核心区别
对比项	    Bagging	        Boosting
代表模型	随机森林	GBDT、XGBoost、LightGBM
训练方式	多个模型并行训练	多个模型串行训练
每个模型关系	相对独立	后一个依赖前一个
核心思想	多模型投票或平均	后一个模型修正前一个错误
主要作用	降低方差，提高稳定性	降低偏差，提高拟合能力
是否容易并行	容易	相对困难
过拟合风险	相对较低	参数不当时可能过拟合

一句话总结：

Bagging 是多个模型一起投票；
Boosting 是多个模型接力改错。
六、GBDT 是什么？

GBDT 全称是：

Gradient Boosting Decision Tree

中文可以叫：

梯度提升决策树

它由两部分组成：

GB：Gradient Boosting，梯度提升
DT：Decision Tree，决策树

所以 GBDT 可以理解为：

用很多棵决策树，按照 Boosting 的方式，一棵一棵地修正前面模型的错误。
七、GBDT 的核心思想

GBDT 和随机森林最大的区别是：

随机森林：很多棵树并行训练，然后投票。
GBDT：树是一棵接一棵训练，后面的树修正前面的错误。

举一个回归任务例子。

假设真实房价是：

真实房价 = 300 万

第 1 棵树预测：

第 1 棵树预测 = 260 万

还差：

误差 = 300 - 260 = 40 万

第 2 棵树就重点学习这个误差：

第 2 棵树预测修正值 = 30 万

此时总预测变成：

260 + 30 = 290 万

还差：

300 - 290 = 10 万

第 3 棵树继续修正：

第 3 棵树预测修正值 = 8 万

最终预测：

260 + 30 + 8 = 298 万

越来越接近真实值。

这就是 GBDT 的直观思想：

不是一棵树直接解决问题，而是很多棵树逐步修正错误。
八、GBDT 为什么叫“梯度提升”？

第 4 天你学习过梯度下降。

你知道：

梯度可以告诉模型，参数应该往哪个方向调整，损失函数会下降。

GBDT 中的“梯度”可以先这样理解：

模型根据当前预测错误，找到下一棵树应该重点修正的方向。

也就是说，GBDT 每一轮都会问：

当前模型还有哪些地方预测不好？
下一棵树应该怎么修正，才能让损失函数下降？

所以 GBDT 的训练过程大致是：

先有一个初始预测
    ↓
计算当前预测误差
    ↓
训练一棵新树去拟合误差修正方向
    ↓
把新树加入模型
    ↓
重新计算误差
    ↓
继续训练下一棵树
九、随机森林和 GBDT 的区别
对比项	随机森林 Random Forest	GBDT
集成方式	Bagging	Boosting
树之间关系	相互独立	后一棵树依赖前一棵树
训练方式	并行	串行
最终结果	多棵树投票/平均	多棵树结果累加
主要优势	稳定、不容易过拟合	拟合能力强、效果好
主要风险	可能欠拟合	参数不当容易过拟合
常用场景	稳定基线模型	表格数据强模型

一句话总结：

随机森林是多棵树“各自判断后投票”；
GBDT 是多棵树“接力修正错误”。
十、XGBoost 是什么？

XGBoost 全称是：

Extreme Gradient Boosting

它可以理解为：

GBDT 的工程增强版和正则化增强版。

传统 GBDT 已经很强，但 XGBoost 在很多方面做了优化。

它的核心特点包括：

加入正则化，减少过拟合；
支持二阶梯度信息，优化更精细；
支持缺失值处理；
支持并行优化；
训练速度和效果通常优于传统 GBDT；
在很多 Kaggle 和工业表格数据任务中表现很好。

入门阶段你可以先这样理解：

XGBoost = 更强、更稳定、更工程化的 GBDT。
十一、LightGBM 是什么？

LightGBM 是微软开源的梯度提升框架。

它也是 GBDT 思想的增强版本。

它的特点是：

训练速度快；
内存占用低；
适合大规模数据；
支持类别特征处理；
在工业项目中非常常用。

入门阶段你可以先这样理解：

LightGBM = 更快、更省内存、更适合大数据表格任务的 GBDT。
十二、GBDT、XGBoost、LightGBM 的关系
模型	简单理解
GBDT	原始梯度提升决策树思想
XGBoost	强化版 GBDT，注重效果、正则化和工程优化
LightGBM	高效版 GBDT，注重速度、内存和大规模数据

实际项目中常见选择：

数据量小、中等：RandomForest、GBDT、XGBoost 都可以试
数据量较大：LightGBM 常用
想要强基线：XGBoost、LightGBM 常用
想要简单稳定：RandomForest 常用
十三、今日案例：对比三类树模型

今天使用 Iris 鸢尾花数据集，对比：

DecisionTreeClassifier
RandomForestClassifier
GradientBoostingClassifier

其中：

DecisionTreeClassifier：单棵决策树
RandomForestClassifier：Bagging 思想
GradientBoostingClassifier：Boosting 思想
十四、今日完整代码

新建文件：

day10_gbdt_xgboost_lightgbm_intro.py

代码如下：

"""
第 10 天：GBDT、XGBoost、LightGBM 思想入门。

本案例使用 sklearn 自带的 Iris 鸢尾花数据集，
重点对比三种树模型：

1. DecisionTreeClassifier：单棵决策树
2. RandomForestClassifier：随机森林，Bagging 思想
3. GradientBoostingClassifier：梯度提升树，Boosting 思想

注意：
本代码主要使用 sklearn 内置的 GradientBoostingClassifier。
XGBoost 和 LightGBM 属于第三方库，今天先重点理解思想，
后续真实项目中可以再安装 xgboost、lightgbm 使用。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class TreeModelComparison:
    """
    树模型对比实验类。

    :param test_size: 测试集比例
    :param random_state: 随机种子，保证实验结果可复现
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化树模型对比实验类。

        :param test_size: 测试集比例，例如 0.2 表示 20% 数据作为测试集
        :param random_state: 随机种子，用于保证每次运行结果一致
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

        self.models = {}
        self.results = []
        self.result_df = None

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

    def build_models(self):
        """
        构建需要对比的模型。

        DecisionTreeClassifier:
            单棵决策树。

        RandomForestClassifier:
            随机森林，属于 Bagging 集成学习。

        GradientBoostingClassifier:
            梯度提升树，属于 Boosting 集成学习。

        :return: 无返回值
        """
        self.models = {
            "DecisionTree": DecisionTreeClassifier(
                max_depth=3,
                random_state=self.random_state
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=self.random_state
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state
            )
        }

    def train_and_evaluate_models(self):
        """
        训练并评估所有模型。

        :return: 对比结果 DataFrame
        """
        self.results = []

        for model_name, model in self.models.items():
            print(f"\n========== 正在训练模型：{model_name} ==========")

            model.fit(self.X_train, self.y_train)

            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)

            train_accuracy = accuracy_score(self.y_train, train_pred)
            test_accuracy = accuracy_score(self.y_test, test_pred)

            self.results.append({
                "model": model_name,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy
            })

            print(f"{model_name} 训练集 Accuracy：{train_accuracy:.4f}")
            print(f"{model_name} 测试集 Accuracy：{test_accuracy:.4f}")

            print("\n混淆矩阵：")
            print(confusion_matrix(self.y_test, test_pred))

            print("\n分类报告：")
            print(classification_report(
                self.y_test,
                test_pred,
                target_names=self.target_names
            ))

        self.result_df = pd.DataFrame(self.results)
        return self.result_df

    def show_comparison_result(self):
        """
        输出模型对比结果。

        :return: 无返回值
        """
        print("\n========== 模型准确率对比 ==========")
        print(self.result_df)

    def plot_model_comparison(self):
        """
        绘制不同模型训练集和测试集准确率对比图。

        :return: 无返回值
        """
        x = np.arange(len(self.result_df["model"]))
        width = 0.35

        plt.figure(figsize=(9, 5))

        plt.bar(
            x - width / 2,
            self.result_df["train_accuracy"],
            width,
            label="Train Accuracy"
        )

        plt.bar(
            x + width / 2,
            self.result_df["test_accuracy"],
            width,
            label="Test Accuracy"
        )

        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.title("DecisionTree vs RandomForest vs GradientBoosting")
        plt.xticks(x, self.result_df["model"])
        plt.ylim(0.8, 1.05)
        plt.legend()
        plt.show()

    def show_feature_importance(self):
        """
        输出每个模型的特征重要性。

        说明：
        决策树、随机森林、GBDT 都支持 feature_importances_。
        这个属性可以帮助我们观察模型认为哪些特征更重要。

        :return: 无返回值
        """
        print("\n========== 特征重要性对比 ==========")

        for model_name, model in self.models.items():
            if not hasattr(model, "feature_importances_"):
                continue

            importance_df = pd.DataFrame({
                "feature": self.feature_names,
                "importance": model.feature_importances_
            })

            importance_df = importance_df.sort_values(
                by="importance",
                ascending=False
            )

            print(f"\n----- {model_name} 特征重要性 -----")
            print(importance_df)

    def predict_single_sample(self, sample: np.ndarray):
        """
        使用所有模型预测单条样本。

        :param sample: 单条样本，格式为 [[花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度]]
        :return: 无返回值
        """
        print("\n========== 单条样本预测对比 ==========")
        print("输入样本：", sample)

        for model_name, model in self.models.items():
            pred_id = model.predict(sample)[0]
            pred_name = self.target_names[pred_id]

            print(f"{model_name} 预测类别 ID：{pred_id}，类别名称：{pred_name}")

    def run(self):
        """
        运行完整树模型对比流程。

        :return: 无返回值
        """
        print("========== 第 10 天：GBDT、XGBoost、LightGBM 思想 ==========")

        self.load_data()
        self.build_dataframe()
        self.show_data_info()

        self.split_data()
        self.build_models()

        self.train_and_evaluate_models()
        self.show_comparison_result()
        self.show_feature_importance()

        sample = np.array([[6.0, 3.0, 4.8, 1.8]])
        self.predict_single_sample(sample)

        self.plot_model_comparison()


class GradientBoostingParameterExperiment:
    """
    GradientBoosting 参数实验类。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化 GradientBoosting 参数实验类。

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

    def compare_learning_rate_and_estimators(self):
        """
        对比不同 learning_rate 和 n_estimators 组合下的模型效果。

        learning_rate:
            每棵树对最终模型的贡献权重。

        n_estimators:
            树的数量。

        一般来说：
            learning_rate 越小，通常需要更多树；
            learning_rate 越大，树少时也能快速拟合，但过大可能不稳定。

        :return: 实验结果 DataFrame
        """
        param_list = [
            {"learning_rate": 0.01, "n_estimators": 50},
            {"learning_rate": 0.01, "n_estimators": 100},
            {"learning_rate": 0.01, "n_estimators": 200},
            {"learning_rate": 0.1, "n_estimators": 50},
            {"learning_rate": 0.1, "n_estimators": 100},
            {"learning_rate": 0.1, "n_estimators": 200},
            {"learning_rate": 0.3, "n_estimators": 50},
            {"learning_rate": 0.3, "n_estimators": 100},
            {"learning_rate": 0.3, "n_estimators": 200},
        ]

        results = []

        for params in param_list:
            model = GradientBoostingClassifier(
                learning_rate=params["learning_rate"],
                n_estimators=params["n_estimators"],
                max_depth=3,
                random_state=self.random_state
            )

            model.fit(self.X_train, self.y_train)

            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)

            train_accuracy = accuracy_score(self.y_train, train_pred)
            test_accuracy = accuracy_score(self.y_test, test_pred)

            result = {
                "learning_rate": params["learning_rate"],
                "n_estimators": params["n_estimators"],
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy
            }

            results.append(result)

            print(
                f"learning_rate={params['learning_rate']}，"
                f"n_estimators={params['n_estimators']}，"
                f"训练集准确率={train_accuracy:.4f}，"
                f"测试集准确率={test_accuracy:.4f}"
            )

        self.result_df = pd.DataFrame(results)
        return self.result_df

    def run(self):
        """
        运行 GradientBoosting 参数实验。

        :return: 无返回值
        """
        print("\n========== GradientBoosting 参数实验 ==========")

        self.prepare_data()
        self.compare_learning_rate_and_estimators()

        print("\n========== 参数实验结果 ==========")
        print(self.result_df)


def main():
    """
    主函数。

    :return: 无返回值
    """
    comparison = TreeModelComparison(
        test_size=0.2,
        random_state=42
    )
    comparison.run()

    experiment = GradientBoostingParameterExperiment(
        test_size=0.2,
        random_state=42
    )
    experiment.run()


if __name__ == "__main__":
    main()
十五、代码重点解释
1. 单棵决策树
DecisionTreeClassifier(
    max_depth=3,
    random_state=self.random_state
)

含义：

只训练一棵决策树；
树最大深度为 3；
模型可解释性强，但稳定性相对较弱。
2. 随机森林
RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=self.random_state
)

含义：

训练 100 棵决策树；
每棵树相对独立；
最终投票决定类别；
属于 Bagging 思想。
3. GradientBoostingClassifier
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=self.random_state
)

含义：

训练 100 棵提升树；
每棵树按照顺序训练；
后一棵树修正前一阶段的错误；
属于 Boosting 思想。
十六、GBDT 中几个重要参数
1. n_estimators
n_estimators=100

表示树的数量。

在 GBDT 中，树是逐步累加的。

一般来说：

树太少：可能欠拟合；
树太多：可能过拟合，也会增加训练时间。
2. learning_rate
learning_rate=0.1

表示每棵树对最终模型的贡献大小。

可以理解为：

每一棵新树修正错误时，修正力度有多大。

如果学习率小：
较谨慎；
通常需要更多树
每棵树修正得比；
模型可能更稳。

如果学习率大：

每棵树修正力度更大；
训练可能更快；
但过大可能导致模型不稳定或过拟合。
3. max_depth
max_depth=3

表示每棵树的最大深度。

在 GBDT 中，通常不需要特别深的树。

很多时候使用浅树即可，例如：

max_depth=2
max_depth=3
max_depth=4

原因是：

GBDT 不是靠一棵复杂树解决问题；
而是靠很多棵简单树逐步修正错误。
十七、运行后重点观察什么？

运行代码后，重点观察 5 个地方。

1. 三个模型的准确率对比

你会看到类似：

DecisionTree       train_accuracy=0.9667    test_accuracy=0.9667
RandomForest       train_accuracy=0.9667    test_accuracy=0.9667
GradientBoosting   train_accuracy=1.0000    test_accuracy=0.9667

Iris 数据集比较简单，所以三者差距可能不大。

但要注意观察：

训练集准确率是否明显高于测试集准确率；
如果训练集明显高，测试集没有同步提升，可能存在过拟合倾向。
2. GradientBoosting 是否更容易拟合训练集

你可能会看到：

GradientBoosting 训练集准确率更高

这说明：

Boosting 模型拟合能力更强。

但要注意：

拟合能力强不等于泛化能力一定强；
还要看测试集效果。
3. 混淆矩阵

观察每个模型的混淆矩阵：

哪些类别容易被混淆？
不同模型错的是不是同一类？

在 Iris 中，通常：

setosa 最容易识别；
versicolor 和 virginica 更容易混淆。
4. 特征重要性

三个模型都会输出特征重要性。

你要观察：

DecisionTree 认为哪些特征重要？
RandomForest 认为哪些特征重要？
GradientBoosting 认为哪些特征重要？

通常 Iris 数据中：

petal length
petal width

会比花萼相关特征更重要。

5. learning_rate 和 n_estimators 的关系

参数实验中你会看到：

learning_rate=0.01 时，可能需要更多 n_estimators；
learning_rate=0.1 通常比较平衡；
learning_rate=0.3 可能更快拟合，但也可能不稳定。

这就是 GBDT 调参中非常重要的一组关系：

learning_rate 越小，通常 n_estimators 要越大。
十八、GBDT 和随机森林如何选择？
1. 想要简单稳定

优先试：

RandomForest

适合：

快速建立稳定基线；
不想调太多参数；
想先看特征重要性；
数据量不是特别大。
2. 想要更强效果

可以试：

GBDT
XGBoost
LightGBM

适合：

追求更高准确率；
表格数据任务；
有时间调参；
希望模型拟合能力更强。
3. 工业项目常见顺序

很多项目可以按这个顺序尝试：

LogisticRegression
    ↓
DecisionTree
    ↓
RandomForest
    ↓
GradientBoosting
    ↓
XGBoost / LightGBM

原因是：

先从简单模型建立基线；
再逐步使用更复杂、更强的模型；
比较每一步是否真的带来提升。
十九、XGBoost 和 LightGBM 简单使用模板

今天代码中没有强制安装第三方库，但你可以先知道它们的基本写法。

1. 安装 XGBoost
pip install xgboost

简单模板：

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
2. 安装 LightGBM
pip install lightgbm

简单模板：

from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
二十、今日练习
练习 1：解释 Bagging 和 Boosting

请用自己的话解释：

Bagging 和 Boosting 有什么区别？

参考方向：

Bagging 是多个模型并行训练，最终投票或平均；
Boosting 是多个模型串行训练，后一个模型修正前一个模型的错误。
练习 2：解释随机森林和 GBDT

请回答：

随机森林为什么是 Bagging？
GBDT 为什么是 Boosting？
练习 3：修改 learning_rate

把代码中的：

learning_rate=0.1

分别改成：

learning_rate=0.01
learning_rate=0.05
learning_rate=0.2
learning_rate=0.5

观察：

训练集准确率是否变化？
测试集准确率是否变化？
是否出现训练集很高但测试集没有提升？
练习 4：修改 n_estimators

把：

n_estimators=100

分别改成：

n_estimators=10
n_estimators=50
n_estimators=200
n_estimators=500

观察：

树越多是否一定更好？
训练时间是否增加？
测试集准确率是否一直提高？
练习 5：替换数据集

把 Iris 数据集换成乳腺癌数据集：

from sklearn.datasets import load_breast_cancer

self.dataset = load_breast_cancer()

然后对比：

DecisionTree
RandomForest
GradientBoosting

观察：

哪个模型测试集准确率最高？
哪个模型训练集准确率最高？
是否有模型出现过拟合倾向？
哪些特征重要性最高？
二十一、今日学习检查表
检查项	是否掌握
能理解 Bagging 的思想	
能理解 Boosting 的思想	
能区分随机森林和 GBDT	
能理解 GBDT 是逐步修正错误	
能理解 XGBoost 是增强版 GBDT	
能理解 LightGBM 更适合大规模数据	
能使用 GradientBoostingClassifier	
能理解 learning_rate 的作用	
能理解 n_estimators 的作用	
能对比决策树、随机森林和 GBDT	
二十二、今日总结

第 10 天最重要的是这句话：

随机森林是多棵树并行投票，GBDT 是多棵树串行修正错误。

你现在应该能理解：

RandomForestClassifier(
    n_estimators=100,
    max_depth=3
)

背后的含义是：

训练很多棵相对独立的决策树；
最终用投票方式得到结果；
属于 Bagging。

也能理解：

GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

背后的含义是：

一棵树接一棵树训练；
后一棵树修正前面模型的错误；
多个弱模型累加成强模型；
属于 Boosting。

今天还要记住：

GBDT 是思想基础；
XGBoost 是更强的工程增强版；
LightGBM 是更快、更适合大规模数据的增强版。