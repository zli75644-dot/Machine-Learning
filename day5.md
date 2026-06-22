今天进入机器学习中的第一个经典分类模型：逻辑回归。

注意一个容易误解的点：

逻辑回归虽然名字里有“回归”，但它主要用于分类任务。

第 3 天的线性回归用于预测连续值，例如：

房价 = 280 万
销量 = 1500 件
温度 = 31.5 ℃

第 5 天的逻辑回归用于预测类别，例如：

是否患病：是 / 否
是否垃圾邮件：是 / 否
是否会流失：是 / 否
是否会购买：是 / 否
一、今日学习目标

今天需要掌握 8 个内容：

理解分类任务和回归任务的区别；
理解逻辑回归为什么用于分类；
理解线性分数 z；
理解 Sigmoid 函数；
理解概率输出；
理解分类阈值 threshold；
理解交叉熵损失的基本思想；
使用 sklearn 完成一个二分类任务。
二、从线性回归到逻辑回归

第 3 天我们学过线性回归：

y = w1*x1 + w2*x2 + ... + b

它直接输出一个连续数值。

例如：

输入：面积、房间数、楼层
输出：房价 280 万

但是分类任务不一样。

例如判断一个用户是否会购买商品：

输入：浏览次数、加购次数、停留时长、历史消费金额
输出：购买 / 不购买

模型不能直接输出：

购买 = 280

这没有意义。

所以逻辑回归做了一个转换：

第 1 步：先像线性回归一样计算一个分数 z
第 2 步：把 z 放进 Sigmoid 函数
第 3 步：得到 0 到 1 之间的概率
第 4 步：根据概率判断类别
三、逻辑回归的核心公式
1. 先计算线性分数 z

逻辑回归第一步和线性回归很像：

z = w1*x1 + w2*x2 + ... + b

例如：

z = 浏览次数*w1 + 加购次数*w2 + 停留时长*w3 + 历史消费金额*w4 + b

这个 z 可以理解为模型算出来的一个“倾向分数”。

但是 z 可能是任意数：

-10、-2、0、3、8、100

它还不能直接当作概率。

2. 再使用 Sigmoid 函数转换成概率

Sigmoid 函数是逻辑回归的核心。

它的作用是：

把任意大小的 z，压缩到 0 到 1 之间。

也就是说：

z 很大     → Sigmoid(z) 接近 1
z 等于 0   → Sigmoid(z) = 0.5
z 很小     → Sigmoid(z) 接近 0

所以逻辑回归最终输出的是：

属于正类的概率

例如：

模型输出 0.82

可以理解为：

模型认为这个样本属于正类的概率是 82%
四、什么是正类和负类？

在二分类任务中，通常会把类别编码成：

0：负类
1：正类

例如医疗检测任务：

0：良性
1：恶性

用户购买预测任务：

0：不会购买
1：会购买

垃圾邮件检测任务：

0：正常邮件
1：垃圾邮件

逻辑回归输出的是：

样本属于类别 1 的概率

例如：

predict_proba = 0.87

表示：

模型认为该样本属于类别 1 的概率是 87%
五、分类阈值 threshold

逻辑回归输出的是概率，但是最终还要变成类别。

最常见的阈值是：

threshold = 0.5

判断规则是：

概率 >= 0.5 → 判断为 1
概率 < 0.5  → 判断为 0

例如：

模型输出概率	最终类别
0.91	1
0.73	1
0.51	1
0.49	0
0.22	0
0.05	0

但是阈值不是固定永远只能是 0.5。

在一些业务场景中，阈值可以调整。

例如医疗疾病筛查中，宁愿多报一些疑似病例，也不希望漏掉真正有风险的人，这时候可能会降低阈值：

threshold = 0.3

这样更多样本会被判断为正类。

六、逻辑回归和线性回归的区别
对比项	                线性回归	        逻辑回归
任务类型	            回归任务            分类任务
输出结果	            连续数值	        类别概率
常见输出            	房价、销量、温度	是/否、0/1
核心公式	            y = wx + b	        sigmoid(wx + b)
常见损失	            MSE	                交叉熵损失
sklearn 模型    	LinearRegression	    LogisticRegression

一句话总结：

线性回归直接输出数值；
逻辑回归先输出概率，再根据阈值转成类别。
七、什么是交叉熵损失？

线性回归常用 MSE，因为它预测的是连续数值。

逻辑回归常用交叉熵损失，因为它预测的是概率。

你现在先这样理解：

真实类别是 1，模型预测概率越接近 1，损失越小；
真实类别是 1，模型预测概率越接近 0，损失越大；

真实类别是 0，模型预测概率越接近 0，损失越小；
真实类别是 0，模型预测概率越接近 1，损失越大。

举例：

真实标签	模型预测属于 1 的概率	模型表现
1	        0.95	                很好
1	        0.60	                一般
1       	0.10	                很差
0	        0.05	                很好
0	        0.40	                一般
0	        0.90	                很差

逻辑回归训练的目标就是：

让正确类别对应的概率越来越高。
八、今日案例：乳腺癌二分类

今天使用 sklearn 自带的乳腺癌数据集。

任务是：

根据细胞特征，判断肿瘤是恶性还是良性。

它是一个二分类任务。

数据标签大致可以理解为：

0：恶性 malignant
1：良性 benign

注意：不同数据集的 0 和 1 代表什么，要看数据集说明，不能想当然。

九、安装依赖
pip install numpy pandas scikit-learn matplotlib
十、今日完整代码

新建文件：

day05_logistic_regression.py

代码如下：

"""
第 5 天：逻辑回归 Logistic Regression。

本案例使用 sklearn 自带的乳腺癌数据集，完成一个二分类任务。

核心目标：
1. 理解逻辑回归用于分类任务
2. 理解 predict 和 predict_proba 的区别
3. 理解准确率、混淆矩阵、分类报告
4. 理解模型输出概率后如何根据阈值转成类别
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class BreastCancerLogisticRegression:
    """
    乳腺癌二分类逻辑回归模型。

    :param test_size: 测试集比例
    :param random_state: 随机种子，保证每次划分数据结果一致
    :param max_iter: 逻辑回归最大迭代次数
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42, max_iter: int = 1000):
        """
        初始化乳腺癌二分类逻辑回归模型。

        :param test_size: 测试集比例，例如 0.2 表示 20% 数据作为测试集
        :param random_state: 随机种子，保证实验结果可复现
        :param max_iter: 模型最大迭代次数，避免模型未收敛
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state
        self.max_iter = max_iter

        self.dataset = None
        self.feature_names = None
        self.target_names = None

        self.df = None
        self.X = None
        self.y = None

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
        将数据集转换为 pandas DataFrame，方便查看数据。

        :return: 数据集 DataFrame
        """
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df["label_id"] = self.y
        self.df["label_name"] = self.df["label_id"].apply(lambda label_id: self.target_names[label_id])

        return self.df

    def show_basic_info(self):
        """
        打印数据集基本信息。

        :return: 无返回值
        """
        print("========== 数据集基本信息 ==========")
        print("特征数量：", len(self.feature_names))
        print("特征数据形状：", self.X.shape)
        print("标签数据形状：", self.y.shape)
        print("类别名称：", self.target_names)

        print("\n========== 前 5 行数据 ==========")
        print(self.df.head())

        print("\n========== 类别数量统计 ==========")
        print(self.df["label_name"].value_counts())

    def split_data(self):
        """
        划分训练集和测试集。

        stratify=self.y 表示按照标签比例进行分层抽样，
        让训练集和测试集中的类别比例尽量一致。

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

        逻辑回归对特征尺度比较敏感。
        如果不同特征的数值范围差异很大，模型训练可能不稳定。
        StandardScaler 会将特征转换为均值约为 0、标准差约为 1 的形式。

        注意：
        1. 训练集使用 fit_transform
        2. 测试集只能使用 transform
        这样可以避免测试集信息泄露到训练阶段。

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

        predict 返回最终类别。
        predict_proba 返回属于每个类别的概率。

        :return: 无返回值
        """
        self.y_pred = self.model.predict(self.X_test_scaled)

        proba_all_classes = self.model.predict_proba(self.X_test_scaled)

        # 二分类中，第 1 列通常表示类别 1 的概率
        self.y_proba = proba_all_classes[:, 1]

    def evaluate_model(self):
        """
        评估模型效果。

        :return: 无返回值
        """
        accuracy = accuracy_score(self.y_test, self.y_pred)

        print("\n========== 模型评估结果 ==========")
        print(f"准确率 accuracy：{accuracy:.4f}")

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
        打印部分测试样本的真实类别、预测类别和预测概率。

        :param rows: 展示前多少行结果
        :return: 无返回值
        """
        result_df = pd.DataFrame({
            "真实类别ID": self.y_test,
            "预测类别ID": self.y_pred,
            "预测为类别1的概率": self.y_proba
        })

        result_df["真实类别名称"] = result_df["真实类别ID"].apply(lambda label_id: self.target_names[label_id])
        result_df["预测类别名称"] = result_df["预测类别ID"].apply(lambda label_id: self.target_names[label_id])
        result_df["是否预测正确"] = result_df["真实类别ID"] == result_df["预测类别ID"]

        print("\n========== 测试集预测详情 ==========")
        print(result_df.head(rows))

    def show_model_parameters(self):
        """
        查看逻辑回归模型学习到的参数。

        coef_ 表示每个特征对应的权重。
        intercept_ 表示偏置项。

        :return: 无返回值
        """
        print("\n========== 模型参数 ==========")
        print("权重矩阵形状：", self.model.coef_.shape)
        print("偏置项：", self.model.intercept_)

        coef_df = pd.DataFrame({
            "feature": self.feature_names,
            "weight": self.model.coef_[0]
        })

        coef_df["abs_weight"] = coef_df["weight"].abs()
        coef_df = coef_df.sort_values(by="abs_weight", ascending=False)

        print("\n========== 权重绝对值最大的前 10 个特征 ==========")
        print(coef_df.head(10))

    def predict_with_threshold(self, threshold: float = 0.5):
        """
        使用自定义阈值进行分类。

        默认情况下，概率 >= 0.5 判断为类别 1。
        这里可以手动修改阈值，观察预测结果变化。

        :param threshold: 分类阈值
        :return: 自定义阈值下的预测类别
        """
        custom_pred = (self.y_proba >= threshold).astype(int)

        accuracy = accuracy_score(self.y_test, custom_pred)

        print(f"\n========== 自定义阈值预测：threshold={threshold} ==========")
        print(f"准确率 accuracy：{accuracy:.4f}")
        print("混淆矩阵：")
        print(confusion_matrix(self.y_test, custom_pred))

        return custom_pred

    def predict_single_sample(self, sample_index: int = 0):
        """
        从测试集中取一条样本，演示单条预测过程。

        :param sample_index: 测试集样本索引
        :return: 无返回值
        """
        sample = self.X_test[sample_index: sample_index + 1]
        sample_scaled = self.scaler.transform(sample)

        pred_id = self.model.predict(sample_scaled)[0]
        pred_proba = self.model.predict_proba(sample_scaled)[0]

        print("\n========== 单条样本预测 ==========")
        print("真实类别ID：", self.y_test[sample_index])
        print("真实类别名称：", self.target_names[self.y_test[sample_index]])
        print("预测类别ID：", pred_id)
        print("预测类别名称：", self.target_names[pred_id])
        print("预测为各类别的概率：", pred_proba)

    def run(self):
        """
        运行完整逻辑回归分类流程。

        :return: 无返回值
        """
        print("========== 第 5 天：逻辑回归二分类 ==========")

        self.load_data()
        self.build_dataframe()
        self.show_basic_info()

        self.split_data()
        self.scale_features()
        self.train_model()

        self.predict_test_data()
        self.evaluate_model()

        self.show_prediction_detail(rows=10)
        self.show_model_parameters()

        self.predict_with_threshold(threshold=0.5)
        self.predict_with_threshold(threshold=0.3)
        self.predict_with_threshold(threshold=0.7)

        self.predict_single_sample(sample_index=0)


def main():
    """
    主函数。

    :return: 无返回值
    """
    demo = BreastCancerLogisticRegression(
        test_size=0.2,
        random_state=42,
        max_iter=1000
    )
    demo.run()


if __name__ == "__main__":
    main()
十一、代码重点解释
1. 加载数据
self.dataset = load_breast_cancer()

self.X = self.dataset.data
self.y = self.dataset.target

这里：

X = 细胞相关特征
y = 肿瘤类别
2. 创建逻辑回归模型
self.model = LogisticRegression(max_iter=self.max_iter)

max_iter 是最大迭代次数。

如果模型训练时出现类似警告：

ConvergenceWarning

通常说明模型没有完全收敛，可以适当增大 max_iter。

3. 为什么要标准化？

代码中有这一步：

self.X_train_scaled = self.scaler.fit_transform(self.X_train)
self.X_test_scaled = self.scaler.transform(self.X_test)

这是标准化。

因为逻辑回归对特征尺度比较敏感。

例如有两个特征：

特征 A：范围 0 到 1
特征 B：范围 1000 到 100000

如果不处理，数值范围大的特征可能会影响模型训练稳定性。

所以我们使用：

StandardScaler()

把不同特征转换到相近的数值尺度。

注意：

训练集：fit_transform
测试集：transform

原因是：

标准化参数只能从训练集中学习；
测试集不能参与训练阶段的数据处理规则生成。

否则就会产生数据泄露。

4. predict 和 predict_proba 的区别
predict
self.y_pred = self.model.predict(self.X_test_scaled)

返回最终类别：

0 或 1
predict_proba
proba_all_classes = self.model.predict_proba(self.X_test_scaled)

返回每个类别的概率。

例如：

[0.12, 0.88]

表示：

属于类别 0 的概率是 12%
属于类别 1 的概率是 88%

在二分类中，我们常取类别 1 的概率：

self.y_proba = proba_all_classes[:, 1]
5. 自定义阈值

默认逻辑回归大致等价于：

类别 1 的概率 >= 0.5，预测为 1；
类别 1 的概率 < 0.5，预测为 0。

代码中我们手动测试了不同阈值：

self.predict_with_threshold(threshold=0.5)
self.predict_with_threshold(threshold=0.3)
self.predict_with_threshold(threshold=0.7)

你会看到阈值变了，预测结果也会变化。

十二、运行后重点观察什么？

运行代码后，你重点看 5 个地方。

1. 数据形状
特征数据形状：(569, 30)
标签数据形状：(569,)

表示：

共有 569 条样本；
每条样本有 30 个特征。
2. 类别名称
类别名称：['malignant' 'benign']

表示：

0：malignant，恶性
1：benign，良性
3. 准确率

你可能看到类似：

准确率 accuracy：0.98 左右

表示测试集中大部分样本预测正确。

4. 混淆矩阵

输出类似：

[[40  2]
 [ 1 71]]

可以这样看：

左上：真实为 0，预测为 0
右上：真实为 0，预测为 1
左下：真实为 1，预测为 0
右下：真实为 1，预测为 1

混淆矩阵第 6 天会详细讲，今天先看懂基本结构即可。

5. 预测概率

你会看到类似：

预测为类别1的概率：0.9832

这说明模型认为该样本属于类别 1 的概率非常高。

十三、逻辑回归的训练过程

逻辑回归训练过程和第 4 天的梯度下降思想一致。

整体流程是：

初始化 w 和 b
    ↓
计算线性分数 z = wx + b
    ↓
使用 Sigmoid 得到概率
    ↓
计算交叉熵损失
    ↓
根据梯度更新 w 和 b
    ↓
重复多轮
    ↓
得到最终分类模型

所以你可以发现：

线性回归和逻辑回归都要学习 w 和 b；
区别在于逻辑回归多了一步 Sigmoid，把线性输出变成概率。
十四、逻辑回归适合什么场景？

逻辑回归适合很多二分类任务。

例如：

用户是否会购买
用户是否会流失
订单是否异常
邮件是否垃圾邮件
广告是否会点击
贷款是否违约
商品是否会爆款
评论是否为差评

它的优点是：

模型简单；
训练速度快；
可解释性较强；
适合作为分类任务的基线模型。

它的不足是：

对复杂非线性关系表达能力有限；
需要较好的特征工程；
特征尺度对训练有影响；
对异常值可能比较敏感。
十五、今天必须掌握的核心代码

今天最重要的是下面几行：

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

y_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

它们的含义是：

创建逻辑回归模型；
训练分类模型；
预测最终类别；
预测属于类别 1 的概率；
计算分类准确率。
十六、今日练习
练习 1：解释逻辑回归

请用自己的话解释：

为什么逻辑回归名字里有“回归”，但它主要用于分类？

参考方向：

它先计算类似线性回归的 wx+b；
然后用 Sigmoid 转成概率；
最后根据概率判断类别。
练习 2：解释 predict 和 predict_proba

请回答：

model.predict() 返回什么？
model.predict_proba() 返回什么？
练习 3：修改测试集比例

把：

test_size=0.2

改成：

test_size=0.3

观察：

训练集数量是否变化？
测试集数量是否变化？
准确率是否变化？
练习 4：修改阈值

观察代码中的：

self.predict_with_threshold(threshold=0.3)
self.predict_with_threshold(threshold=0.7)

请思考：

threshold 降低后，预测为类别 1 的样本会变多还是变少？
threshold 提高后，预测为类别 1 的样本会变多还是变少？

答案：

threshold 降低，预测为类别 1 的样本通常会变多；
threshold 提高，预测为类别 1 的样本通常会变少。
练习 5：关闭标准化试试看

把：

self.scale_features()
以及训练和预测中使用的：

self.X_train_scaled
self.X_test_scaled

尝试替换成原始的：

self.X_train
self.X_test

观察是否出现警告、准确率是否变化。

这个练习的目的不是让你记住结论，而是让你感受：

特征尺度会影响模型训练。
十七、今日学习检查表
检查项	是否掌握
能区分回归任务和分类任务	
能理解逻辑回归主要用于分类	
能理解 z = wx + b	
能理解 Sigmoid 把 z 转成概率	
能理解概率大于阈值才判断为类别 1	
能理解 predict 返回类别	
能理解 predict_proba 返回概率	
能理解标准化的作用	
能运行逻辑回归二分类代码	
能观察不同阈值对预测结果的影响	
十八、今日总结
第 5 天最重要的是这句话：

逻辑回归 = 线性组合 wx+b + Sigmoid 概率转换 + 阈值分类。

你现在应该能理解：

线性回归：预测连续数值；
逻辑回归：预测属于某个类别的概率，再转成分类结果。
今日核心流程：

加载数据
    ↓
划分训练集和测试集
    ↓
标准化特征
    ↓
训练 LogisticRegression
    ↓
predict 预测类别
    ↓
predict_proba 预测概率
    ↓
accuracy_score 评估准确率