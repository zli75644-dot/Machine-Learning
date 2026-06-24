已思考 7s
第 8 天：决策树 Decision Tree

今天学习一个非常适合入门理解的机器学习模型：决策树。

决策树的核心思想非常像人做判断：

如果 花瓣长度 <= 2.45，那么判断为 setosa；
否则继续判断 花瓣宽度 <= 1.75；
如果满足，则判断为 versicolor；
否则判断为 virginica。

也就是说，决策树就是通过一系列 if-else 判断规则 来完成分类或回归任务。

一、今日学习目标

今天你需要掌握 8 个内容：

理解决策树的基本思想；
理解决策树中的根节点、内部节点、叶子节点；
理解分类树和回归树；
理解基尼系数 Gini；
理解信息熵 Entropy；
理解决策树为什么容易过拟合；
掌握 max_depth 的作用；
使用 sklearn 训练并可视化一棵决策树。
二、什么是决策树？

决策树是一种树形结构的机器学习模型。

它通过不断提问，把数据一步一步分开。

例如判断一个人是否适合打篮球：

身高是否大于 185？
    是 → 体重是否大于 75？
        是 → 可能适合
        否 → 继续观察
    否 → 可能不适合

这就是一棵简单的决策树。

在机器学习中，决策树会自动从数据中学习这些规则。

三、决策树的结构

一棵决策树通常由三类节点组成。

1. 根节点 Root Node

根节点是决策树的第一个判断条件。

例如：

petal length <= 2.45

它是整棵树最开始分裂的位置。

2. 内部节点 Internal Node

内部节点是中间的判断条件。

例如：

petal width <= 1.75

它继续把数据分成更小的子集。

3. 叶子节点 Leaf Node

叶子节点是最终输出结果的地方。

例如：

class = setosa
class = versicolor
class = virginica

叶子节点不再继续分裂，直接给出预测类别。

四、决策树是怎么分类的？

假设现在有一朵花：

花萼长度 = 6.0
花萼宽度 = 3.0
花瓣长度 = 4.8
花瓣宽度 = 1.8

决策树可能这样判断：

花瓣长度 <= 2.45？
    否

花瓣宽度 <= 1.75？
    否

预测类别：virginica

所以，决策树的预测过程就是：

从根节点开始；
根据条件向左或向右走；
一直走到叶子节点；
叶子节点的类别就是预测结果。
五、决策树如何选择分裂条件？

决策树训练时要解决一个问题：

先用哪个特征来分？
用什么阈值来分？

例如 Iris 数据集有 4 个特征：

sepal length
sepal width
petal length
petal width

模型要自动判断：

到底是先按花萼长度分？
还是先按花瓣长度分？
阈值应该选 2.45，还是 3.15？

决策树选择分裂条件的标准通常有两个：

Gini 基尼系数
Entropy 信息熵
六、Gini 基尼系数

Gini 用来衡量一个节点的“混杂程度”。

如果一个节点里全是同一类样本，那么这个节点很纯，Gini 很小。

如果一个节点里各种类别都有，那么这个节点很混乱，Gini 较大。

Gini 公式是：

其中：

C：类别数量
p_i：第 i 类样本在当前节点中的比例

你可以这样理解：

Gini 越小，节点越纯；
Gini 越大，节点越混杂。

决策树训练时，会优先选择让子节点更纯的分裂方式。

七、Entropy 信息熵

信息熵也是衡量混乱程度的指标。

Entropy 公式是：

你可以这样理解：

Entropy 越小，说明节点越纯；
Entropy 越大，说明节点越混乱。

在 sklearn 中，决策树默认使用：

criterion="gini"

你也可以改成：

criterion="entropy"
八、Gini 和 Entropy 的区别
对比项	            Gini	            Entropy
中文名	            基尼系数	        信息熵
作用	            衡量节点纯度	    衡量节点混乱程度
sklearn 参数	    criterion="gini"	criterion="entropy"
计算速度	        一般更快	        稍慢
实际效果	        常用默认选择	    也很常用

入门阶段你先记住：

Gini 和 Entropy 都是为了帮助决策树选择更好的分裂条件。
九、决策树为什么容易过拟合？

决策树如果不加限制，会不断分裂，直到训练数据几乎被完全分开。

这样会导致一个问题：

模型在训练集上表现很好；
但在测试集或新数据上表现可能变差。

这就是过拟合。

例如：

训练集准确率：100%
测试集准确率：80%

说明模型可能把训练数据中的细节、噪声、偶然情况都记住了。

十、如何控制决策树复杂度？

常用参数有 4 个。

1. max_depth

限制树的最大深度。

max_depth=3

表示树最多只能分裂 3 层。

这是最常用的防止过拟合参数。

2. min_samples_split

限制一个节点至少要有多少样本，才允许继续分裂。

min_samples_split=5

表示一个节点至少有 5 个样本，才能继续分裂。

3. min_samples_leaf

限制叶子节点至少要有多少样本。

min_samples_leaf=2

表示每个叶子节点至少要有 2 个样本。

4. criterion

选择分裂标准。

criterion="gini"
criterion="entropy"
十一、今日案例：Iris 鸢尾花决策树分类

今天继续使用 Iris 数据集。

任务是：

根据花萼长度、花萼宽度、花瓣长度、花瓣宽度，判断鸢尾花类别。

类别包括：

setosa
versicolor
virginica
十二、今日完整代码

新建文件：

day08_decision_tree.py

代码如下：

"""
第 8 天：决策树 Decision Tree。

本案例使用 sklearn 自带的 Iris 鸢尾花数据集，
演示决策树分类模型的完整流程。

核心目标：
1. 理解决策树的基本思想
2. 理解根节点、内部节点、叶子节点
3. 理解 max_depth 对模型复杂度的影响
4. 对比不同树深度下的训练集和测试集准确率
5. 可视化决策树结构
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class IrisDecisionTreeClassifier:
    """
    鸢尾花决策树分类器。

    :param test_size: 测试集比例
    :param random_state: 随机种子，保证每次划分数据结果一致
    :param max_depth: 决策树最大深度
    :param criterion: 节点分裂标准，可选 gini 或 entropy
    :return: 无返回值
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        max_depth: int = 3,
        criterion: str = "gini"
    ):
        """
        初始化鸢尾花决策树分类器。

        :param test_size: 测试集比例，例如 0.2 表示 20% 数据作为测试集
        :param random_state: 随机种子，用于保证实验结果可复现
        :param max_depth: 决策树最大深度，用于控制模型复杂度
        :param criterion: 决策树分裂标准，默认使用 gini
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state
        self.max_depth = max_depth
        self.criterion = criterion

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

        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

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

    def train_model(self):
        """
        训练决策树分类模型。

        :return: 无返回值
        """
        self.model.fit(self.X_train, self.y_train)
        print("\n决策树模型训练完成。")
        print(f"当前 criterion：{self.criterion}")
        print(f"当前 max_depth：{self.max_depth}")

    def predict_test_data(self):
        """
        使用测试集进行预测。

        :return: 测试集预测结果
        """
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def evaluate_model(self):
        """
        评估决策树分类模型。

        :return: 无返回值
        """
        train_pred = self.model.predict(self.X_train)

        train_accuracy = accuracy_score(self.y_train, train_pred)
        test_accuracy = accuracy_score(self.y_test, self.y_pred)

        print("\n========== 模型评估结果 ==========")
        print(f"训练集 Accuracy：{train_accuracy:.4f}")
        print(f"测试集 Accuracy：{test_accuracy:.4f}")

        print("\n========== 混淆矩阵 ==========")
        print(confusion_matrix(self.y_test, self.y_pred))

        print("\n========== 分类报告 ==========")
        print(classification_report(
            self.y_test,
            self.y_pred,
            target_names=self.target_names
        ))

    def show_tree_rules(self):
        """
        以文本形式输出决策树学到的规则。

        :return: 无返回值
        """
        tree_rules = export_text(
            self.model,
            feature_names=list(self.feature_names)
        )

        print("\n========== 决策树规则 ==========")
        print(tree_rules)

    def show_feature_importance(self):
        """
        输出决策树中的特征重要性。

        :return: 无返回值
        """
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="importance",
            ascending=False
        )

        print("\n========== 特征重要性 ==========")
        print(importance_df)

    def visualize_tree(self):
        """
        可视化决策树结构。

        :return: 无返回值
        """
        plt.figure(figsize=(14, 8))

        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=self.target_names,
            filled=True,
            rounded=True
        )

        plt.title("Decision Tree Visualization")
        plt.show()

    def predict_single_sample(self, sample: np.ndarray):
        """
        预测单条鸢尾花样本。

        :param sample: 单条样本，格式为 [[花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度]]
        :return: 预测类别 ID 和类别名称
        """
        pred_id = self.model.predict(sample)[0]
        pred_name = self.target_names[pred_id]

        return pred_id, pred_name

    def run(self):
        """
        运行完整决策树分类流程。

        :return: 无返回值
        """
        print("========== 第 8 天：决策树分类 ==========")

        self.load_data()
        self.build_dataframe()
        self.show_data_info()

        self.split_data()
        self.train_model()
        self.predict_test_data()

        self.evaluate_model()
        self.show_tree_rules()
        self.show_feature_importance()

        sample = np.array([[6.0, 3.0, 4.8, 1.8]])
        pred_id, pred_name = self.predict_single_sample(sample)

        print("\n========== 单条样本预测 ==========")
        print("输入样本：", sample)
        print("预测类别 ID：", pred_id)
        print("预测类别名称：", pred_name)

        self.visualize_tree()


class DecisionTreeDepthExperiment:
    """
    决策树深度对比实验类。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化决策树深度对比实验类。

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

    def compare_different_depths(self, depth_values: list):
        """
        比较不同 max_depth 下的训练集和测试集准确率。

        :param depth_values: 决策树最大深度列表
        :return: 实验结果 DataFrame
        """
        results = []

        for depth in depth_values:
            model = DecisionTreeClassifier(
                max_depth=depth,
                random_state=self.random_state
            )

            model.fit(self.X_train, self.y_train)

            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)

            train_accuracy = accuracy_score(self.y_train, train_pred)
            test_accuracy = accuracy_score(self.y_test, test_pred)

            results.append({
                "max_depth": depth,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy
            })

            print(
                f"max_depth={depth}，"
                f"训练集准确率={train_accuracy:.4f}，"
                f"测试集准确率={test_accuracy:.4f}"
            )

        self.result_df = pd.DataFrame(results)
        return self.result_df

    def plot_depth_accuracy_curve(self):
        """
        绘制 max_depth 与训练集、测试集准确率关系图。

        :return: 无返回值
        """
        plt.figure(figsize=(8, 5))

        plt.plot(
            self.result_df["max_depth"],
            self.result_df["train_accuracy"],
            marker="o",
            label="Train Accuracy"
        )

        plt.plot(
            self.result_df["max_depth"],
            self.result_df["test_accuracy"],
            marker="o",
            label="Test Accuracy"
        )

        plt.xlabel("max_depth")
        plt.ylabel("Accuracy")
        plt.title("Decision Tree Depth vs Accuracy")
        plt.xticks(self.result_df["max_depth"])
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        """
        运行决策树深度对比实验。

        :return: 无返回值
        """
        print("\n========== 决策树 max_depth 对比实验 ==========")

        self.prepare_data()

        depth_values = [1, 2, 3, 4, 5, 6, 7, 8]
        self.compare_different_depths(depth_values=depth_values)

        print("\n========== max_depth 对比结果 ==========")
        print(self.result_df)

        self.plot_depth_accuracy_curve()


def main():
    """
    主函数。

    :return: 无返回值
    """
    tree_demo = IrisDecisionTreeClassifier(
        test_size=0.2,
        random_state=42,
        max_depth=3,
        criterion="gini"
    )

    tree_demo.run()

    depth_experiment = DecisionTreeDepthExperiment(
        test_size=0.2,
        random_state=42
    )

    depth_experiment.run()


if __name__ == "__main__":
    main()
十三、代码重点解释
1. 创建决策树模型
self.model = DecisionTreeClassifier(
    criterion=self.criterion,
    max_depth=self.max_depth,
    random_state=self.random_state
)

这里最重要的是两个参数：

criterion="gini"
max_depth=3

含义是：

使用 Gini 作为分裂标准；
决策树最大深度限制为 3。
2. 决策树通常不需要标准化

你会发现，今天代码没有使用：

StandardScaler()

原因是：

KNN、逻辑回归这类模型容易受特征尺度影响；
决策树主要根据特征阈值切分数据，对标准化不敏感。

例如：

面积 <= 100
收入 <= 5000
花瓣长度 <= 2.45

决策树关注的是“是否小于某个阈值”，不是直接计算距离。

所以决策树通常不强制要求标准化。

3. 查看决策树规则
tree_rules = export_text(
    self.model,
    feature_names=list(self.feature_names)
)

这段代码会输出类似：

|--- petal length (cm) <= 2.45
|   |--- class: 0
|--- petal length (cm) >  2.45
|   |--- petal width (cm) <= 1.75
|   |   |--- class: 1
|   |--- petal width (cm) >  1.75
|   |   |--- class: 2

这说明模型学到的规则比较直观。

4. 查看特征重要性
self.model.feature_importances_

决策树可以输出特征重要性。

例如可能看到：

petal length (cm)    0.55
petal width (cm)     0.42
sepal length (cm)    0.03
sepal width (cm)     0.00

说明：

花瓣长度和花瓣宽度对分类最重要；
花萼相关特征在这棵树中作用较小。
5. 可视化决策树
plot_tree(
    self.model,
    feature_names=self.feature_names,
    class_names=self.target_names,
    filled=True,
    rounded=True
)

这段代码会画出整棵决策树。

图中你重点看：

每个节点的判断条件；
每个节点的 gini；
每个节点的 samples；
每个节点的 value；
每个叶子节点的 class。
十四、决策树图怎么看？

可视化图中通常会显示类似内容：

petal length (cm) <= 2.45
gini = 0.667
samples = 120
value = [40, 40, 40]
class = setosa

逐个解释：

1. 判断条件
petal length (cm) <= 2.45

表示当前节点按照花瓣长度是否小于等于 2.45 来分裂。

2. gini
gini = 0.667

表示当前节点的混杂程度。

越小越纯。

3. samples
samples = 120

表示当前节点包含 120 条训练样本。

4. value
value = [40, 40, 40]

表示当前节点中 3 个类别的样本数量。

例如：

setosa：40 条
versicolor：40 条
virginica：40 条
5. class
class = setosa

表示当前节点如果作为预测结果，会预测成哪个类别。

一般是当前节点中数量最多的类别。

十五、运行后重点观察什么？

运行代码后，重点看 4 个地方。

1. 训练集准确率和测试集准确率

例如：

训练集 Accuracy：0.9667
测试集 Accuracy：0.9667

如果训练集很高，测试集明显低很多，例如：

训练集 Accuracy：1.0000
测试集 Accuracy：0.8000

这可能说明：

模型过拟合了。
2. 决策树规则

观察输出的规则：

模型最先用哪个特征分裂？
哪些特征被使用了？
哪些特征没有被使用？

你很可能会发现：

Iris 分类中，petal length 和 petal width 非常重要。
3. 特征重要性

重点看：

importance 最大的特征是谁？
importance 为 0 的特征有没有？

如果某个特征重要性为 0，说明这棵树没有使用它进行分裂。

4. max_depth 对比实验

你会看到不同深度下：

训练集准确率
测试集准确率

随着 max_depth 增大，通常：

训练集准确率会上升；
测试集准确率不一定一直上升。

这就是过拟合现象的直观表现。

十六、决策树和 KNN 的区别
对比项	KNN	决策树
核心思想	最近邻投票	if-else 规则判断
是否依赖距离	是	否
是否需要标准化	通常需要	通常不需要
训练过程	主要保存数据	学习分裂规则
可解释性	一般	很强
预测速度	数据量大时较慢	快
容易过拟合吗	K 太小时容易	树太深时容易

一句话总结：

KNN 是看邻居；
决策树是学规则。
十七、决策树的优缺点
1. 优点
容易理解；
可解释性强；
可以可视化；
不太需要标准化；
可以处理分类和回归任务；
能输出特征重要性。
2. 缺点
容易过拟合；
对数据细微变化比较敏感；
单棵树稳定性不如随机森林；
如果树太深，泛化能力可能下降。

所以，实际项目中经常使用决策树的集成版本：

随机森林 Random Forest
GBDT
XGBoost
LightGBM
CatBoost

第 9 天就会学习随机森林。

十八、今日练习
练习 1：解释决策树

请用自己的话解释：

决策树是如何完成分类的？

参考方向：

它通过一系列特征判断条件，把样本一步步分到不同分支，最后走到叶子节点，叶子节点的类别就是预测结果。
练习 2：修改 max_depth

把代码中的：

max_depth=3

分别改成：

max_depth=1
max_depth=2
max_depth=4
max_depth=8

观察：

训练集准确率怎么变？
测试集准确率怎么变？
树的结构是否变复杂？
练习 3：修改 criterion

把：

criterion="gini"

改成：

criterion="entropy"

观察：

树的规则是否变化？
准确率是否变化？
特征重要性是否变化？
练习 4：解释树图中的字段

查看可视化图，找一个节点解释：

判断条件是什么？
gini 是多少？
samples 是多少？
value 表示什么？
class 是什么？
练习 5：预测新样本

修改单条预测样本：

sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample = np.array([[6.0, 3.0, 4.8, 1.8]])
sample = np.array([[6.5, 3.0, 5.2, 2.0]])

观察预测类别分别是什么。

十九、今日学习检查表
检查项	是否掌握
能理解决策树的基本思想	
能理解根节点、内部节点、叶子节点	
能理解 Gini 是衡量节点纯度	
能理解 Entropy 是衡量混乱程度	
能使用 DecisionTreeClassifier	
能理解 max_depth 的作用	
能知道树太深容易过拟合	
能看懂 export_text 输出的规则	
能看懂决策树可视化图	
能解释特征重要性	
二十、今日总结

第 8 天最重要的是这句话：

决策树就是通过一系列 if-else 判断规则，把样本一步步分到叶子节点，并用叶子节点的类别作为预测结果。

你现在应该能理解下面代码：

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

背后的含义是：

创建一棵最大深度为 3 的决策树；
模型自动学习特征分裂规则；
用这些规则对新样本进行分类。

今天还要记住：

决策树优点是直观、可解释、能可视化；
缺点是容易过拟合；
控制 max_depth 是防止过拟合的重要方法。