第 6 天：分类评估指标

前面第 5 天你已经完成了逻辑回归二分类任务。

你已经会用：

model.fit(X_train, y_train)
model.predict(X_test)
accuracy_score(y_test, y_pred)

但是今天要解决一个更重要的问题：

模型准确率高，就一定说明模型好吗？

答案是：

不一定。

尤其在医疗检测、风控、欺诈识别、异常检测、垃圾邮件识别等任务中，只看准确率很容易误判模型效果。

一、今日学习目标

今天需要掌握 6 个核心指标：

Accuracy 准确率；
Confusion Matrix 混淆矩阵；
Precision 精确率；
Recall 召回率；
F1-score；
classification_report 分类报告。
今天你要形成一个意识：

分类模型不能只看准确率，要结合混淆矩阵、精确率、召回率和 F1-score 一起判断。
问题：准确率和精确率的区别，F1得分是啥？
二、为什么准确率不一定可靠？

假设现在做一个疾病筛查任务。

有 1000 个人，其中：

990 人没有病
10 人有病

现在有一个很“偷懒”的模型，它对所有人都预测：

没有病

那么结果是：

990 人预测正确
10 人预测错误

准确率是：

990 / 1000 = 99%

看起来准确率很高。

但是这个模型真正有用吗？

没用。

因为它把所有真正有病的人都漏掉了。

所以在分类任务中，我们不能只看：

预测对了多少

还要看：

哪些类别预测错了？
把正类漏掉了多少？
预测为正类的样本有多少是真的？
三、混淆矩阵 Confusion Matrix

混淆矩阵是分类评估中最核心的工具。

对于二分类任务，混淆矩阵长这样：

真实情况 / 预测结果	预测为 0	预测为 1
真实为 0	        TN	        FP
真实为 1	        FN	        TP

这四个缩写很重要。

1. TP：True Positive
真实是 1，预测也是 1。

中文可以理解为：

真正例

例如：

真实有病，模型也预测有病。

这是正确预测。

2. TN：True Negative
真实是 0，预测也是 0。

中文可以理解为：

真反例

例如：

真实没病，模型也预测没病。

这是正确预测。

3. FP：False Positive
真实是 0，但预测成 1。

中文可以理解为：

假正例

例如：

真实没病，但模型预测有病。

这叫误报。

4. FN：False Negative
真实是 1，但预测成 0。

中文可以理解为：

假反例

例如：

真实有病，但模型预测没病。

这叫漏报。

四、用疾病检测理解四个结果

假设：

1 = 有病
0 = 没病

那么：

缩写	含义	医疗例子
TP	真实为 1，预测为 1	有病，预测有病
TN	真实为 0，预测为 0	没病，预测没病
FP	真实为 0，预测为 1	没病，误判有病
FN	真实为 1，预测为 0	有病，漏判没病

在医疗检测中，最危险的一般是：

FN：有病但被预测为没病。

因为这会导致真正有风险的人被漏掉。

五、Accuracy 准确率

准确率表示：

所有样本中，预测正确的比例。

公式是：

Accuracy = 预测正确的数量 / 总样本数量

也就是：

准确率适合类别比较均衡的任务。

例如：

猫狗分类
鸢尾花分类
普通图片分类

但是如果类别严重不均衡，准确率可能会骗人。

六、Precision 精确率

精确率关注的是：

模型预测为正类的样本中，有多少是真的正类。

公式是：

通俗理解：

模型说这些人有病，那么其中有多少人真的有病？

精确率高说明：

模型只要预测为正类，通常比较靠谱。
精确率适合关注什么场景？

精确率适合关注“误报成本高”的场景。

例如：

垃圾邮件检测
金融风控拦截
内容违规检测
广告欺诈判定

比如正常邮件被误判成垃圾邮件，这会影响用户体验。

这时候就希望：

模型预测为垃圾邮件时，尽量是真的垃圾邮件。
七、Recall 召回率

召回率关注的是：

真实正类中，有多少被模型找出来了。

公式是：

通俗理解：

所有真正有病的人里面，模型成功发现了多少？

召回率高说明：

模型漏掉的正类比较少。
召回率适合关注什么场景？

召回率适合关注“漏报成本高”的场景。

例如：

疾病筛查
欺诈交易检测
安全入侵检测
高危订单识别
设备故障预警

这些场景中，真正危险的是：

明明有问题，却被模型判断为没问题。

所以更关注 Recall。

八、Precision 和 Recall 的区别

你可以这样记：

Precision：模型预测为正类的结果，有多少是真的？
Recall：所有真实正类中，模型找回了多少？

举个例子。

假设有 100 个病人，其中真正有病的有 10 人。

模型预测有病的有 8 人，其中 6 人是真的有病。

那么：

TP = 6
FP = 2
FN = 4

Precision 是：

6 / (6 + 2) = 0.75

意思是：

模型预测有病的 8 人中，有 75% 真的有病。

Recall 是：

6 / (6 + 4) = 0.60

意思是：

真正有病的 10 人中，模型只找出了 60%。
九、F1-score

有时候 Precision 和 Recall 一个高一个低。

例如：

Precision 很高，但 Recall 很低；
Recall 很高，但 Precision 很低。

这时候可以看 F1-score。

F1-score 是 Precision 和 Recall 的综合指标。

F1-score 越高，说明 Precision 和 Recall 的综合表现越好。

注意：

F1-score 不是简单平均值，而是调和平均。

它的特点是：

只要 Precision 或 Recall 其中一个很低，F1-score 也会被拉低。
十、分类报告 classification_report

sklearn 中可以直接输出分类报告：

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

它会输出：

precision
recall
f1-score
support

其中：

support = 每个类别在测试集中的真实样本数量

例如：

              precision    recall  f1-score   support

   malignant       0.95      0.98      0.96        42
      benign       0.99      0.97      0.98        72

你要这样看：

malignant 这一行表示恶性类别的评估结果；
benign 这一行表示良性类别的评估结果。
十一、今日完整代码

新建文件：

day06_classification_metrics.py

代码如下：

"""
第 6 天：分类评估指标。

本案例继续使用 sklearn 自带的乳腺癌数据集，
重点学习分类模型评估指标：

1. Accuracy 准确率
2. Confusion Matrix 混淆矩阵
3. Precision 精确率
4. Recall 召回率
5. F1-score
6. classification_report 分类报告

核心目标：
不要只看准确率，要学会综合判断分类模型效果。
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class ClassificationMetricsDemo:
    """
    分类评估指标演示类。

    :param test_size: 测试集比例
    :param random_state: 随机种子，保证实验结果可复现
    :param max_iter: 逻辑回归最大迭代次数
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42, max_iter: int = 1000):
        """
        初始化分类评估指标演示类。

        :param test_size: 测试集比例，例如 0.2 表示 20% 数据作为测试集
        :param random_state: 随机种子，用于保证每次运行结果一致
        :param max_iter: 逻辑回归最大迭代次数
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state
        self.max_iter = max_iter

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
        self.model = LogisticRegression(max_iter=self.max_iter)

        self.X_train_scaled = None
        self.X_test_scaled = None

        self.y_pred = None
        self.y_proba = None

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

        :return: 数据集 DataFrame
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
        print("类别名称：", self.target_names)

        print("\n========== 类别数量统计 ==========")
        print(self.df["label_name"].value_counts())

        print("\n========== 标签 ID 与类别名称对应关系 ==========")
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

        print("\n========== 数据划分结果 ==========")
        print("训练集特征形状：", self.X_train.shape)
        print("测试集特征形状：", self.X_test.shape)
        print("训练集标签形状：", self.y_train.shape)
        print("测试集标签形状：", self.y_test.shape)

    def scale_features(self):
        """
        对特征进行标准化处理。

        训练集使用 fit_transform。
        测试集使用 transform。

        :return: 无返回值
        """
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train_model(self):
        """
        训练逻辑回归模型。

        :return: 无返回值
        """
        self.model.fit(self.X_train_scaled, self.y_train)
        print("\n模型训练完成。")

    def predict_test_data(self):
        """
        使用测试集进行预测。

        :return: 无返回值
        """
        self.y_pred = self.model.predict(self.X_test_scaled)
        self.y_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]

    def evaluate_basic_metrics(self):
        """
        计算基础分类指标。

        :return: 无返回值
        """
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)

        print("\n========== 基础分类评估指标 ==========")
        print(f"Accuracy 准确率：{accuracy:.4f}")
        print(f"Precision 精确率：{precision:.4f}")
        print(f"Recall 召回率：{recall:.4f}")
        print(f"F1-score：{f1:.4f}")

    def evaluate_confusion_matrix(self):
        """
        计算并解释混淆矩阵。

        sklearn 的 confusion_matrix 输出格式：
        [[TN, FP],
         [FN, TP]]

        :return: 无返回值
        """
        matrix = confusion_matrix(self.y_test, self.y_pred)

        tn, fp, fn, tp = matrix.ravel()

        print("\n========== 混淆矩阵 ==========")
        print(matrix)

        print("\n========== 混淆矩阵拆解 ==========")
        print(f"TN 真反例：真实为 0，预测为 0，数量：{tn}")
        print(f"FP 假正例：真实为 0，预测为 1，数量：{fp}")
        print(f"FN 假反例：真实为 1，预测为 0，数量：{fn}")
        print(f"TP 真正例：真实为 1，预测为 1，数量：{tp}")

    def evaluate_classification_report(self):
        """
        输出 sklearn 分类报告。

        :return: 无返回值
        """
        report = classification_report(
            self.y_test,
            self.y_pred,
            target_names=self.target_names
        )

        print("\n========== classification_report 分类报告 ==========")
        print(report)

    def show_prediction_detail(self, rows: int = 15):
        """
        展示部分预测结果。

        :param rows: 展示前多少行
        :return: 无返回值
        """
        result_df = pd.DataFrame({
            "真实类别ID": self.y_test,
            "预测类别ID": self.y_pred,
            "预测为类别1的概率": self.y_proba
        })

        result_df["真实类别名称"] = result_df["真实类别ID"].apply(
            lambda label_id: self.target_names[label_id]
        )
        result_df["预测类别名称"] = result_df["预测类别ID"].apply(
            lambda label_id: self.target_names[label_id]
        )
        result_df["是否预测正确"] = result_df["真实类别ID"] == result_df["预测类别ID"]

        print("\n========== 部分预测结果 ==========")
        print(result_df.head(rows))

    def evaluate_with_custom_threshold(self, threshold: float):
        """
        使用自定义阈值重新生成预测类别，并计算指标。

        :param threshold: 分类阈值
        :return: 无返回值
        """
        custom_pred = (self.y_proba >= threshold).astype(int)

        accuracy = accuracy_score(self.y_test, custom_pred)
        precision = precision_score(self.y_test, custom_pred)
        recall = recall_score(self.y_test, custom_pred)
        f1 = f1_score(self.y_test, custom_pred)
        matrix = confusion_matrix(self.y_test, custom_pred)

        print(f"\n========== 自定义阈值 threshold={threshold} ==========")
        print(f"Accuracy 准确率：{accuracy:.4f}")
        print(f"Precision 精确率：{precision:.4f}")
        print(f"Recall 召回率：{recall:.4f}")
        print(f"F1-score：{f1:.4f}")
        print("混淆矩阵：")
        print(matrix)

    def compare_thresholds(self):
        """
        比较不同阈值下的分类指标变化。

        :return: 无返回值
        """
        thresholds = [0.3, 0.5, 0.7]

        for threshold in thresholds:
            self.evaluate_with_custom_threshold(threshold=threshold)

    def run(self):
        """
        运行完整分类评估流程。

        :return: 无返回值
        """
        print("========== 第 6 天：分类评估指标 ==========")

        self.load_data()
        self.build_dataframe()
        self.show_data_info()

        self.split_data()
        self.scale_features()
        self.train_model()
        self.predict_test_data()

        self.evaluate_basic_metrics()
        self.evaluate_confusion_matrix()
        self.evaluate_classification_report()
        self.show_prediction_detail(rows=15)
        self.compare_thresholds()


def main():
    """
    主函数。

    :return: 无返回值
    """
    demo = ClassificationMetricsDemo(
        test_size=0.2,
        random_state=42,
        max_iter=1000
    )

    demo.run()


if __name__ == "__main__":
    main()
十二、代码重点解释
1. 基础指标计算
accuracy = accuracy_score(self.y_test, self.y_pred)
precision = precision_score(self.y_test, self.y_pred)
recall = recall_score(self.y_test, self.y_pred)
f1 = f1_score(self.y_test, self.y_pred)

这里默认把类别 1 当作正类。

在乳腺癌数据集中：

0 -> malignant
1 -> benign

所以默认情况下，precision_score、recall_score、f1_score 计算的是类别 1，也就是 benign 的指标。

如果想关注类别 0，可以写：

precision_score(self.y_test, self.y_pred, pos_label=0)
recall_score(self.y_test, self.y_pred, pos_label=0)
f1_score(self.y_test, self.y_pred, pos_label=0)

这一点非常重要。

2. 混淆矩阵拆解
matrix = confusion_matrix(self.y_test, self.y_pred)

tn, fp, fn, tp = matrix.ravel()

对于二分类，sklearn 的混淆矩阵默认格式是：

[[TN, FP],
 [FN, TP]]

也就是：

第 1 行：真实类别为 0 的样本
第 2 行：真实类别为 1 的样本

第 1 列：预测类别为 0 的样本
第 2 列：预测类别为 1 的样本
3. classification_report
print(classification_report(
    self.y_test,
    self.y_pred,
    target_names=self.target_names
))

这个报告一次性输出：

每个类别的 precision
每个类别的 recall
每个类别的 f1-score
每个类别的 support

它比单独看 accuracy 更全面。

4. 自定义阈值
custom_pred = (self.y_proba >= threshold).astype(int)

含义是：

如果预测为类别 1 的概率 >= threshold，就预测为 1；
否则预测为 0。

当 threshold 变小时：

更容易预测为 1；
类别 1 的 Recall 通常会上升；
但 Precision 可能下降。

当 threshold 变大时：

更不容易预测为 1；
类别 1 的 Precision 可能上升；
但 Recall 可能下降。
十三、运行后重点观察什么？

运行代码后，重点观察 4 个部分。

1. 类别 ID 对应关系

你会看到：

0 -> malignant
1 -> benign

所以在这个数据集中：

类别 0 是恶性
类别 1 是良性

这会影响你对 Precision 和 Recall 的解释。

2. 混淆矩阵

例如输出：

[[40  2]
 [ 1 71]]

含义是：

真实恶性且预测恶性：40
真实恶性但预测良性：2
真实良性但预测恶性：1
真实良性且预测良性：71

在医疗场景中，如果你更关注“恶性是否被发现”，应该重点看类别 0 的 Recall，而不是默认类别 1 的 Recall。

3. 分类报告

你要分别看每一类：

malignant
benign

不要只看最后的 accuracy。

尤其要看：

恶性 malignant 的 recall 是否足够高。

因为真实恶性被预测成良性，风险非常高。

4. 不同阈值下的指标变化

你会看到 threshold 分别为：

0.3
0.5
0.7

时，Precision、Recall、F1-score 和混淆矩阵都会变化。

这说明：

模型输出的是概率，最终分类结果受阈值影响。
十四、重点补充：医疗任务到底该关注哪个类别？

乳腺癌数据集中：

0 = malignant 恶性
1 = benign 良性

但是很多时候我们习惯把：

1 = 有风险
0 = 无风险

所以这里容易混乱。

在这个数据集中，如果你关心“恶性是否被发现”，你应该把 malignant 当成重点类别。

也就是关注：

recall_score(y_test, y_pred, pos_label=0)

可以增加下面这段代码：

def evaluate_malignant_metrics(self):
    """
    单独评估 malignant 恶性类别的指标。

    在 sklearn 乳腺癌数据集中：
    0 表示 malignant
    1 表示 benign

    :return: 无返回值
    """
    malignant_precision = precision_score(
        self.y_test,
        self.y_pred,
        pos_label=0
    )

    malignant_recall = recall_score(
        self.y_test,
        self.y_pred,
        pos_label=0
    )

    malignant_f1 = f1_score(
        self.y_test,
        self.y_pred,
        pos_label=0
    )

    print("\n========== malignant 恶性类别指标 ==========")
    print(f"恶性 Precision：{malignant_precision:.4f}")
    print(f"恶性 Recall：{malignant_recall:.4f}")
    print(f"恶性 F1-score：{malignant_f1:.4f}")

然后在 run() 里面加：

self.evaluate_malignant_metrics()

这样你就能单独观察恶性类别的识别情况。

十五、什么时候关注 Precision？什么时候关注 Recall？
1. 更关注 Precision 的场景

Precision 关注：

模型预测为正类时，到底准不准。

适合误报成本高的场景。

例如：

把正常邮件误判为垃圾邮件；
把正常用户误判为欺诈用户；
把正常内容误判为违规内容；
把正常订单误判为风险订单。

这些场景中，误报太多会影响正常用户。

2. 更关注 Recall 的场景

Recall 关注：

真实正类中，被模型找出来了多少。

适合漏报成本高的场景。

例如：

疾病筛查漏掉患者；
欺诈交易漏掉风险订单；
服务器攻击漏掉异常请求；
设备故障预警漏掉故障信号。

这些场景中，漏报可能造成严重后果。

3. 关注 F1-score 的场景

F1-score 适合：

Precision 和 Recall 都比较重要，需要综合平衡。

例如：

评论情感分类；
用户流失预测；
普通二分类模型对比；
多个模型之间做综合评估。
十六、今天必须掌握的核心代码
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

matrix = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

你需要理解：

accuracy_score：整体预测正确率
precision_score：预测为正类的样本中，有多少是真的
recall_score：真实正类中，有多少被找出来
f1_score：Precision 和 Recall 的综合指标
confusion_matrix：查看预测对错的具体分布
classification_report：完整分类报告
十七、今日练习
练习 1：解释混淆矩阵

请你解释下面这个混淆矩阵：

[[50, 5],
 [3, 42]]

请回答：

TN 是多少？
FP 是多少？
FN 是多少？
TP 是多少？
总共有多少条样本？
预测正确了多少条？
预测错误了多少条？

答案：

TN = 50
FP = 5
FN = 3
TP = 42
总样本 = 50 + 5 + 3 + 42 = 100
预测正确 = 50 + 42 = 92
预测错误 = 5 + 3 = 8
练习 2：手动计算 Accuracy

根据上面的混淆矩阵：

[[50, 5],
 [3, 42]]

计算 Accuracy：

Accuracy = (TP + TN) / 总样本
Accuracy = (42 + 50) / 100
Accuracy = 0.92
练习 3：手动计算 Precision

根据：

TP = 42
FP = 5

计算 Precision：

Precision = TP / (TP + FP)
Precision = 42 / (42 + 5)
Precision = 42 / 47
Precision ≈ 0.8936
练习 4：手动计算 Recall

根据：

TP = 42
FN = 3

计算 Recall：

Recall = TP / (TP + FN)
Recall = 42 / (42 + 3)
Recall = 42 / 45
Recall ≈ 0.9333
练习 5：思考业务场景

请判断下面场景更关注 Precision 还是 Recall：

场景	更关注 Precision / Recall
疾病筛查	Recall
垃圾邮件识别	Precision
欺诈交易检测	Recall
内容违规拦截	Precision
设备故障预警	Recall
正常用户封禁识别	Precision
十八、今日学习检查表
检查项	是否掌握
能理解准确率 Accuracy	
能理解混淆矩阵	
能区分 TP、TN、FP、FN	
能理解 Precision 精确率	
能理解 Recall 召回率	
能理解 F1-score	
能看懂 classification_report	
能理解 support 的含义	
能知道什么时候关注 Precision	
能知道什么时候关注 Recall	
能运行完整分类评估代码	
十九、今日总结

第 6 天最重要的是这句话：

分类模型评估不能只看准确率，必须结合混淆矩阵、Precision、Recall 和 F1-score 一起判断。

你现在应该能看懂：

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

背后的含义：

模型不是简单地“对了多少”，还要看它错在哪里。

今天你尤其要记住：

Precision：预测为正类的样本中，有多少是真的正类。
Recall：真实正类中，有多少被模型找出来。
F1-score：Precision 和 Recall 的综合表现。