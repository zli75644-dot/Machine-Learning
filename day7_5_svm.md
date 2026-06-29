补充课：SVM 支持向量机

也可以叫：

第 7.5 天：SVM 支持向量机
一、今日学习目标

这一节你要掌握：

1. 什么是 SVM；
2. 什么是最大间隔；
3. 什么是支持向量；
4. 什么是线性 SVM；
5. 什么是核函数；
6. C 参数是什么意思；
7. gamma 参数是什么意思；
8. 如何用 sklearn 训练 SVM 分类模型。
二、SVM 是什么？

SVM 全称是：

Support Vector Machine

中文叫：

支持向量机

它主要用于：

分类任务

也可以用于回归任务，但入门阶段先重点学分类。

SVM 的核心思想是：

在不同类别之间找到一条分界线，并且让这条分界线离两边样本尽可能远。
三、用二维图理解 SVM

假设有两类样本：

红色点：类别 0
蓝色点：类别 1

现在要画一条线把它们分开。

可能有很多条线都能分开：

线 A 可以分开
线 B 可以分开
线 C 也可以分开

SVM 选择哪一条？

SVM 会选择：

离两类样本都最远的那条线。

这条线叫：

最大间隔分类超平面

简单说：

SVM 不是随便找一条能分开的线；
而是找一条最安全、最稳的分界线。
四、什么是间隔 Margin？

间隔就是：

分界线到最近样本点的距离。

SVM 希望这个距离越大越好。

例如：

分界线离红色点很近，说明稍微有点扰动就可能分错；
分界线离两边点都比较远，说明分类边界更稳。

SVM 的目标可以理解为：

最大化分类边界和最近样本之间的距离。
五、什么是支持向量？

支持向量就是：

离分类边界最近的那些样本点。

这些点最关键。

因为它们决定了分界线的位置。

你可以这样理解：

普通样本离边界很远，对分界线影响不大；
支持向量离边界最近，真正决定了边界应该怎么画。

这也是为什么它叫：

支持向量机

因为这个模型主要依赖那些关键的“支持向量”。

六、SVM 的核心公式直观理解

线性分类边界可以写成：

你可以这样理解：

w：控制分界线方向；
x：样本特征；
b：偏置；
w^T x + b = 0：分类边界。

如果：

w^T x + b > 0

模型判断为一类。

如果：

w^T x + b < 0

模型判断为另一类。

七、SVM 和逻辑回归有什么区别？
对比项	逻辑回归	SVM
主要任务	分类	分类
输出	概率	类别边界
核心思想	学习类别概率	找最大间隔分界线
是否关注距离边界最近的点	不特别强调	非常强调
是否需要标准化	通常需要	通常需要
可解释性	较强	中等
非线性能力	较弱	可通过核函数增强

一句话：

逻辑回归更像是在估计概率；
SVM 更像是在找一条最稳的分类边界。
八、SVM 和 KNN 有什么区别？
对比项	KNN	SVM
核心思想	看最近邻居投票	找最大间隔边界
是否依赖距离	是	是
是否需要标准化	需要	需要
训练过程	基本保存数据	学习分类边界
预测速度	数据大时慢	通常较快
对异常点	K 小时敏感	C 参数会影响敏感程度
常用场景	小数据直观分类	中小规模分类、高维文本分类
九、什么是核函数？

有些数据不是一条直线能分开的。

例如：

一类点在中间；
另一类点围在外面。

这种情况线性分界线很难分开。

SVM 可以通过核函数把数据映射到更高维空间，让原本不好分的数据变得好分。

常见核函数：

kernel	含义	说明
linear	线性核	适合线性可分数据
rbf	径向基核	最常用，能处理非线性
poly	多项式核	可以学习多项式边界
sigmoid	Sigmoid 核	较少用

入门阶段先记住：

数据简单、特征很多：可以试 linear
不知道用什么：可以先试 rbf
十、SVM 重要参数
1. C 参数
C=1.0

C 控制模型对错误分类的容忍程度。

可以这样理解：

C 大：模型更不愿意分错训练样本，边界更严格，可能过拟合；
C 小：模型允许一些训练样本分错，边界更宽松，可能更稳，但也可能欠拟合。
C 值	模型特点
小 C	间隔更大，容错更强，模型更简单
大 C	更努力分对训练集，模型更复杂
2. gamma 参数

gamma 主要用于 RBF 核。

gamma="scale"

可以这样理解：

gamma 控制一个样本点影响范围有多大。
gamma	含义
gamma 小	单个样本影响范围大，边界更平滑
gamma 大	单个样本影响范围小，边界更复杂，可能过拟合

入门建议：

gamma="scale"

先用默认值。

十一、SVM 为什么必须标准化？

SVM 会计算样本到分类边界的距离。

如果特征尺度差别很大：

年龄：18 到 60
收入：3000 到 50000

收入的数值范围会主导距离计算。

所以 SVM 前通常要做：

StandardScaler()

这一点和 KNN、KMeans、PCA 很像。

十二、SVM 完整代码案例

新建文件：

day07_5_svm_classification.py

代码如下：

"""
补充课：SVM 支持向量机分类。

本案例使用 sklearn 自带的乳腺癌数据集，完成一个二分类任务。

核心目标：
1. 理解 SVM 用于分类任务
2. 理解线性核和 RBF 核
3. 理解 C 参数
4. 理解 gamma 参数
5. 理解为什么 SVM 需要标准化
6. 对比不同 kernel 的分类效果
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class BreastCancerSVMDemo:
    """
    乳腺癌 SVM 分类演示类。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化 SVM 分类演示类。

        :param test_size: 测试集比例，例如 0.2 表示 20% 数据作为测试集
        :param random_state: 随机种子，保证实验结果可复现
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

        self.model = None
        self.y_pred = None

    def load_data(self):
        """
        加载乳腺癌数据集。

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

        :return: 无返回值
        """
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df["label_id"] = self.y
        self.df["label_name"] = self.df["label_id"].apply(
            lambda label_id: self.target_names[label_id]
        )

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

    def build_svm_pipeline(self, kernel: str = "rbf", C: float = 1.0, gamma="scale"):
        """
        构建 SVM Pipeline。

        SVM 对特征尺度敏感，所以需要 StandardScaler。

        :param kernel: 核函数，例如 linear、rbf、poly
        :param C: 正则化参数，C 越大越严格拟合训练集
        :param gamma: RBF 核参数，控制单个样本影响范围
        :return: 无返回值
        """
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel=kernel,
                        C=C,
                        gamma=gamma,
                        probability=True,
                        random_state=self.random_state
                    )
                )
            ]
        )

    def train_model(self):
        """
        训练 SVM 模型。

        :return: 无返回值
        """
        self.model.fit(self.X_train, self.y_train)
        print("\nSVM 模型训练完成。")

    def predict_test_data(self):
        """
        使用测试集进行预测。

        :return: 无返回值
        """
        self.y_pred = self.model.predict(self.X_test)

    def evaluate_model(self):
        """
        评估模型效果。

        :return: 无返回值
        """
        accuracy = accuracy_score(self.y_test, self.y_pred)

        print("\n========== 模型评估结果 ==========")
        print(f"Accuracy 准确率：{accuracy:.4f}")

        print("\n混淆矩阵：")
        print(confusion_matrix(self.y_test, self.y_pred))

        print("\n分类报告：")
        print(classification_report(
            self.y_test,
            self.y_pred,
            target_names=self.target_names
        ))

    def predict_single_sample(self):
        """
        预测单条样本。

        :return: 无返回值
        """
        sample = self.X_test[:1]

        pred_label = int(self.model.predict(sample)[0])
        pred_proba = self.model.predict_proba(sample)[0]

        print("\n========== 单条样本预测 ==========")
        print("预测标签 ID：", pred_label)
        print("预测类别名称：", self.target_names[pred_label])
        print("类别概率：", pred_proba)

    def run(self):
        """
        运行完整 SVM 分类流程。

        :return: 无返回值
        """
        print("========== 补充课：SVM 支持向量机 ==========")

        self.load_data()
        self.build_dataframe()
        self.show_data_info()
        self.split_data()

        self.build_svm_pipeline(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        )

        self.train_model()
        self.predict_test_data()
        self.evaluate_model()
        self.predict_single_sample()


class SVMParameterExperiment:
    """
    SVM 参数对比实验类。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化 SVM 参数实验类。

        :param test_size: 测试集比例
        :param random_state: 随机种子
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state

        dataset = load_breast_cancer()
        self.X = dataset.data
        self.y = dataset.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )

        self.result_df = None

    def compare_kernels(self):
        """
        对比不同 kernel 的 SVM 模型效果。

        :return: 对比结果 DataFrame
        """
        kernels = ["linear", "rbf", "poly"]

        results = []

        print("\n========== SVM kernel 对比 ==========")

        for kernel in kernels:
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        SVC(
                            kernel=kernel,
                            C=1.0,
                            gamma="scale",
                            random_state=self.random_state
                        )
                    )
                ]
            )

            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)

            results.append({
                "kernel": kernel,
                "accuracy": accuracy
            })

            print(f"kernel={kernel}, Accuracy={accuracy:.4f}")

        self.result_df = pd.DataFrame(results)
        return self.result_df

    def compare_c_values(self):
        """
        对比不同 C 值下 RBF-SVM 的效果。

        :return: 对比结果 DataFrame
        """
        c_values = [0.01, 0.1, 1, 10, 100]

        results = []

        print("\n========== SVM C 参数对比 ==========")

        for c_value in c_values:
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        SVC(
                            kernel="rbf",
                            C=c_value,
                            gamma="scale",
                            random_state=self.random_state
                        )
                    )
                ]
            )

            model.fit(self.X_train, self.y_train)

            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)

            train_accuracy = accuracy_score(self.y_train, train_pred)
            test_accuracy = accuracy_score(self.y_test, test_pred)
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

        result_df = pd.DataFrame(results)
        return result_df


def main():
    """
    主函数。

    :return: 无返回值
    """
    demo = BreastCancerSVMDemo(
        test_size=0.2,
        random_state=42
    )

    demo.run()

    experiment = SVMParameterExperiment(
        test_size=0.2,
        random_state=42
    )

    kernel_result_df = experiment.compare_kernels()
    print("\n========== kernel 对比结果 ==========")
    print(kernel_result_df)

    c_result_df = experiment.compare_c_values()
    print("\n========== C 参数对比结果 ==========")
    print(c_result_df)


if __name__ == "__main__":
    main()
十三、代码重点解释
1. 导入 SVM
from sklearn.svm import SVC

SVC 表示：

Support Vector Classification

也就是支持向量分类器。

2. 为什么用 Pipeline？
self.model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf"))
    ]
)

因为 SVM 需要标准化。

Pipeline 可以保证：

训练集上 fit 标准化；
测试集上 transform 标准化；
避免数据泄露；
代码结构更清晰。
3. kernel="rbf"
SVC(kernel="rbf")

表示使用 RBF 核。

它能处理非线性分类边界。

入门阶段常用：

kernel="rbf"
4. probability=True
SVC(probability=True)

默认 SVM 不直接输出概率。

如果你想使用：

predict_proba()

就需要设置：

probability=True

注意：

开启 probability=True 会让训练稍慢一些。
十四、SVM 优缺点
优点
适合中小规模数据；
适合高维特征；
分类边界清晰；
RBF 核可以处理非线性问题；
在传统文本分类中也很常用。
缺点
对特征标准化敏感；
参数 C、gamma 需要调；
数据量很大时训练较慢；
结果不如决策树直观解释；
概率输出不是默认开启。
十五、SVM 适合什么场景？

SVM 适合：

中小规模分类任务；
特征维度比较高的数据；
文本分类；
医学分类；
图像特征分类；
二分类任务。

例如：

垃圾邮件识别；
评论情感分类；
肿瘤良恶性分类；
学生是否高风险；
图片特征分类。
十六、SVM 不太适合什么场景？

SVM 不太适合：

样本量特别大的任务；
需要强业务解释的场景；
类别特别多且数据很大的任务；
实时训练要求很高的任务。

这种时候可以优先考虑：

随机森林；
LightGBM；
深度学习模型。
十七、SVM 应该放进 20 天计划哪里？

更合理的 20 天计划应该改成这样：

第 1 天：机器学习整体认知
第 2 天：sklearn 工具链
第 3 天：线性回归
第 4 天：梯度下降
第 5 天：逻辑回归
第 6 天：分类评估指标
第 7 天：KNN
第 8 天：SVM 支持向量机
第 9 天：决策树
第 10 天：随机森林
第 11 天：GBDT / XGBoost / LightGBM
第 12 天：数据预处理
第 13 天：特征工程
第 14 天：模型选择与交叉验证
第 15 天：过拟合与欠拟合
第 16 天：聚类
第 17 天：PCA 降维
第 18 天：文本机器学习
第 19 天：模型保存与部署
第 20 天：完整项目实战与复盘

也就是说：

SVM 应该插在 KNN 后面、决策树前面。
十八、今日练习
练习 1：解释 SVM

用自己的话解释：

SVM 为什么叫支持向量机？

参考答案：

因为分类边界主要由离边界最近的那些关键样本点决定，这些关键样本点叫支持向量。
练习 2：修改 kernel

把：

kernel="rbf"

分别改成：

kernel="linear"
kernel="poly"

观察 Accuracy 是否变化。

练习 3：修改 C

尝试：

C=0.01
C=0.1
C=1
C=10
C=100

观察：

训练集准确率是否升高？
测试集准确率是否升高？
训练集和测试集差距是否变大？
练习 4：关闭标准化

把 Pipeline 中的：

("scaler", StandardScaler())

去掉，直接训练 SVM。

观察模型效果是否变化。

思考：

为什么 SVM 对特征尺度敏感？
练习 5：加入到第 13 天模型选择代码里

在第 13 天候选模型中加入：

"SVM": Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", C=1.0, gamma="scale"))
    ]
)

然后和：

LogisticRegression
KNN
DecisionTree
RandomForest
GradientBoosting

一起做交叉验证比较。

十九、SVM 总结

SVM 最重要的一句话是：

SVM 是一种寻找最大间隔分类边界的模型，它依靠离边界最近的支持向量来确定分类边界。

你需要记住：

SVM 适合分类任务；
SVM 关注最大间隔；
支持向量是离边界最近的关键样本；
SVM 通常需要标准化；
kernel="linear" 适合线性问题；
kernel="rbf" 适合非线性问题；
C 越大，模型越严格拟合训练集；
gamma 越大，边界越复杂，可能过拟合。