第 11 天：数据预处理 Data Preprocessing

前 10 天你主要学习的是模型：

线性回归
逻辑回归
KNN
决策树
随机森林
GBDT

从第 11 天开始，重点从“模型本身”转向“数据本身”。

在真实机器学习项目中，很多时候不是模型决定上限，而是数据质量决定上限。

你可以记住一句话：

脏数据直接进模型，模型只会学到脏规律。
一、今日学习目标

今天需要掌握 7 个内容：

什么是数据预处理；
缺失值如何处理；
异常值如何处理；
类别变量如何编码；
One-Hot Encoding 是什么；
Label Encoding 是什么；
标准化和归一化的区别。

今天你最终要完成一个目标：

封装一个 DataPreprocessor 数据预处理类。
二、什么是数据预处理？

数据预处理就是在模型训练之前，对原始数据进行清洗、转换和规范化。

原始数据通常会有这些问题：

有缺失值；
有异常值；
有中文类别字段；
有字符串字段；
不同特征数值范围差异很大；
日期字段无法直接进入模型；
某些字段对模型没有意义；
标签列和特征列混在一起。

例如一份学生成绩数据：

姓名	性别	年龄	学习时长	是否挂科
张三	男  	20	    5.5     	否
李四	女	    空	    6.0     	否
王五	男	    200	    0.5     	是
赵六	女	    21	    空	        否

这份数据直接丢给模型是不合适的。

原因是：

年龄有缺失；
年龄 200 明显异常；
性别是字符串；
是否挂科是中文类别；
学习时长有缺失。

所以需要预处理。

三、缺失值处理

缺失值就是数据中没有值。

常见表现：

NaN
None
空字符串
null
未知
-

例如：

年龄	学习时长
20	    5.5
21	    6.0
NaN	    4.0
22	    NaN
1. 删除缺失值

如果缺失数据很少，可以直接删除。

df.dropna()

缺点是：

可能丢失样本；
数据量小时不建议随便删除。
2. 用均值填充

适合数值型字段。

df["age"] = df["age"].fillna(df["age"].mean())

例如年龄缺失，可以用平均年龄填充。

3. 用中位数填充

如果数据中有异常值，中位数比均值更稳。

例如收入数据：

3000, 4000, 5000, 1000000

均值会被 1000000 拉得很高。

这种情况更适合用中位数。

4. 用众数填充

适合类别字段。

例如性别缺失：

df["gender"] = df["gender"].fillna(df["gender"].mode()[0])
四、异常值处理

异常值就是明显不合理的数据。

例如：

年龄 = 200
学习时长 = -5
考试成绩 = 150
商品价格 = -99
身高 = 500cm

异常值不处理，可能会严重影响模型。

1. 业务规则处理

最直接的方法是根据业务常识判断。

例如：

年龄正常范围：0 到 120
考试成绩正常范围：0 到 100
学习时长不能小于 0
商品价格不能小于 0

代码示例：

df = df[(df["age"] >= 0) & (df["age"] <= 120)]
2. 截断处理

不删除样本，而是把异常值限制在合理范围内。

例如：

df["age"] = df["age"].clip(lower=0, upper=120)

含义是：

小于 0 的变成 0；
大于 120 的变成 120。
3. IQR 四分位法

IQR 是常见的异常值检测方法。

IQR = Q3 - Q1

通常认为：

小于 Q1 - 1.5 * IQR 的值是异常值；
大于 Q3 + 1.5 * IQR 的值是异常值。

入门阶段你先理解：

IQR 是一种基于数据分布自动找异常值的方法。
五、类别变量编码

机器学习模型通常不能直接处理字符串。

例如：

男
女
广州
深圳
普通用户
高价值用户

这些都需要转换成数字。

1. Label Encoding

Label Encoding 是把类别转成编号。

例如性别：

性别	编码
男	0
女	1

例如用户等级：

用户等级	编码
低价值	0
普通	1
高价值	2

代码示例：

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["gender_encoded"] = encoder.fit_transform(df["gender"])
2. Label Encoding 的问题

Label Encoding 会引入一种“大小关系”。

例如：

广州 = 0
深圳 = 1
北京 = 2
上海 = 3

模型可能误以为：

上海 > 北京 > 深圳 > 广州

但城市之间本来没有这种大小关系。

所以对于没有顺序关系的类别字段，更常用 One-Hot Encoding。

3. One-Hot Encoding

One-Hot Encoding 会把一个类别字段拆成多个 0/1 字段。

例如城市字段：

city
广州
深圳
北京

转换后：

city_广州	city_深圳	city_北京
1	0	0
0	1	0
0	0	1

这样就不会引入错误的大小关系。

代码示例：

pd.get_dummies(df, columns=["city"])
六、标准化 StandardScaler

标准化是把数据转换成：

均值约为 0；
标准差约为 1。

常用在：

逻辑回归
KNN
SVM
神经网络
线性模型

例如两个特征：

年龄：18 到 60
收入：3000 到 50000

收入数值范围远大于年龄。

如果不处理，模型可能更关注收入。

使用标准化后，不同特征会处在更接近的尺度上。

代码：

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
七、归一化 MinMaxScaler

归一化是把数据压缩到指定范围，通常是 0 到 1。

公式大致是：

新值 = (原值 - 最小值) / (最大值 - 最小值)

例如：

原始值	归一化后
0	    0
50	    0.5
100	    1

代码：

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
八、标准化和归一化的区别
对比项	标准化 StandardScaler	    归一化 MinMaxScaler
结果范围	不固定  	            通常 0 到 1
核心思想	转换为均值 0、标准差 1	按最大最小值压缩
是否容易受异常值影响	相对较小	比较敏感
常用场景	逻辑回归、KNN、SVM	    神经网络、图像像素、特定范围需求
sklearn 类	StandardScaler	        MinMaxScaler

入门阶段建议：

不确定用哪个时，传统机器学习先用 StandardScaler。
九、训练集和测试集预处理的重点

这一点非常重要：

预处理规则只能从训练集中学习，不能从测试集中学习。

错误写法：

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

问题是：

测试集单独 fit_transform，会让测试集生成自己的标准化规则；
这不符合真实预测场景。

正确写法：

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

含义是：

训练集：学习规则并转换；
测试集：使用训练集学到的规则转换。

这个原则适用于：

缺失值填充；
标准化；
归一化；
One-Hot 编码；
特征选择。
十、今日案例：学生是否挂科预测数据预处理

今天构造一个模拟数据集。

任务是：

根据学生信息，预测学生是否挂科。

原始字段：

gender：性别，类别字段
age：年龄，数值字段
study_hours：每日学习时长，数值字段
attendance_rate：出勤率，数值字段
major：专业，类别字段
failed：是否挂科，标签字段

其中故意加入：

缺失值；
异常年龄；
类别变量；
数值尺度差异。
十一、今日完整代码

新建文件：

day11_data_preprocessing.py

代码如下：

"""
第 11 天：数据预处理 Data Preprocessing。

本案例使用一个模拟的学生数据集，演示机器学习项目中常见的数据预处理流程：

1. 构造原始数据
2. 查看缺失值
3. 处理异常值
4. 划分特征 X 和标签 y
5. 划分训练集和测试集
6. 数值特征缺失值填充
7. 类别特征缺失值填充
8. 类别特征 One-Hot 编码
9. 数值特征标准化
10. 训练逻辑回归模型
11. 评估模型效果

重点：
预处理规则只能在训练集上 fit，然后应用到测试集 transform。
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class StudentDataBuilder:
    """
    学生数据构造类。

    :param None: 初始化时不需要外部参数
    :return: 无返回值
    """

    def __init__(self):
        """
        初始化学生数据构造类。

        :return: 无返回值
        """
        self.df = None

    def build_raw_data(self) -> pd.DataFrame:
        """
        构造包含缺失值、异常值和类别字段的模拟学生数据。

        字段说明：
        gender: 性别
        age: 年龄
        study_hours: 每日学习时长
        attendance_rate: 出勤率
        major: 专业
        failed: 是否挂科，Yes 表示挂科，No 表示未挂科

        :return: 原始学生数据 DataFrame
        """
        data = {
            "gender": [
                "男", "女", "男", "女", "男",
                "女", "男", None, "女", "男",
                "女", "男", "女", "男", "女"
            ],
            "age": [
                20, 21, 22, np.nan, 20,
                21, 200, 22, 19, 20,
                21, 23, np.nan, 22, 20
            ],
            "study_hours": [
                1.0, 3.5, 2.0, 5.0, np.nan,
                6.0, 0.5, 4.0, 3.0, 1.5,
                5.5, 2.5, 4.5, np.nan, 6.5
            ],
            "attendance_rate": [
                0.60, 0.90, 0.75, 0.95, 0.50,
                0.98, 0.40, 0.88, np.nan, 0.65,
                0.96, 0.70, 0.92, 0.55, 0.99
            ],
            "major": [
                "物联网", "软件工程", "人工智能", "物联网", "软件工程",
                "人工智能", "物联网", "软件工程", None, "人工智能",
                "物联网", "软件工程", "人工智能", "物联网", "软件工程"
            ],
            "failed": [
                "Yes", "No", "No", "No", "Yes",
                "No", "Yes", "No", "No", "Yes",
                "No", "No", "No", "Yes", "No"
            ]
        }

        self.df = pd.DataFrame(data)
        return self.df


class DataPreprocessor:
    """
    数据预处理类。

    :param numeric_features: 数值特征列名列表
    :param categorical_features: 类别特征列名列表
    :return: 无返回值
    """

    def __init__(self, numeric_features: list, categorical_features: list):
        """
        初始化数据预处理类。

        :param numeric_features: 数值特征列名列表
        :param categorical_features: 类别特征列名列表
        :return: 无返回值
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        self.preprocessor = None
        self.label_encoder = LabelEncoder()

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理异常值。

        当前案例中主要处理 age 年龄字段：
        1. 年龄小于 0 不合理
        2. 年龄大于 120 不合理

        这里采用截断方式，而不是删除样本。

        :param df: 原始数据 DataFrame
        :return: 处理异常值后的 DataFrame
        """
        df = df.copy()

        if "age" in df.columns:
            df["age"] = df["age"].clip(lower=0, upper=120)

        return df

    def build_preprocessor(self):
        """
        构建 sklearn ColumnTransformer 预处理器。

        数值字段处理流程：
        1. 使用中位数填充缺失值
        2. 使用 StandardScaler 标准化

        类别字段处理流程：
        1. 使用众数填充缺失值
        2. 使用 OneHotEncoder 独热编码

        :return: ColumnTransformer 预处理器
        """
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features)
            ]
        )

        return self.preprocessor

    def encode_label(self, y: pd.Series) -> np.ndarray:
        """
        对标签列进行编码。

        例如：
        No -> 0
        Yes -> 1

        :param y: 原始标签列
        :return: 编码后的标签数组
        """
        y_encoded = self.label_encoder.fit_transform(y)
        return y_encoded

    def show_label_mapping(self):
        """
        打印标签编码映射关系。

        :return: 无返回值
        """
        print("\n========== 标签编码映射 ==========")
        for index, class_name in enumerate(self.label_encoder.classes_):
            print(f"{class_name} -> {index}")

    def fit_transform_train(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        在训练集上学习预处理规则，并转换训练集。

        :param X_train: 训练集特征
        :return: 预处理后的训练集特征数组
        """
        if self.preprocessor is None:
            self.build_preprocessor()

        X_train_processed = self.preprocessor.fit_transform(X_train)
        return X_train_processed

    def transform_test(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        使用训练集学到的预处理规则转换测试集。

        注意：
        测试集只能 transform，不能 fit_transform。

        :param X_test: 测试集特征
        :return: 预处理后的测试集特征数组
        """
        X_test_processed = self.preprocessor.transform(X_test)
        return X_test_processed

    def get_processed_feature_names(self) -> list:
        """
        获取预处理后的特征名称。

        数值特征保持原名。
        类别特征经过 OneHotEncoder 后会展开成多个新列。

        :return: 预处理后的特征名称列表
        """
        feature_names = []

        feature_names.extend(self.numeric_features)

        onehot_encoder = (
            self.preprocessor
            .named_transformers_["cat"]
            .named_steps["onehot"]
        )

        categorical_output_names = onehot_encoder.get_feature_names_out(
            self.categorical_features
        )

        feature_names.extend(categorical_output_names.tolist())

        return feature_names


class StudentFailurePredictionDemo:
    """
    学生挂科预测演示类。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.3, random_state: int = 42):
        """
        初始化学生挂科预测演示类。

        :param test_size: 测试集比例，例如 0.3 表示 30% 数据作为测试集
        :param random_state: 随机种子，保证实验结果可复现
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state

        self.df = None

        self.numeric_features = [
            "age",
            "study_hours",
            "attendance_rate"
        ]

        self.categorical_features = [
            "gender",
            "major"
        ]

        self.label_column = "failed"

        self.X = None
        self.y = None
        self.y_encoded = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.X_train_processed = None
        self.X_test_processed = None

        self.preprocessor = DataPreprocessor(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features
        )

        self.model = LogisticRegression(max_iter=1000)
        self.y_pred = None

    def load_data(self):
        """
        加载模拟学生数据。

        :return: 无返回值
        """
        builder = StudentDataBuilder()
        self.df = builder.build_raw_data()

    def show_raw_data(self):
        """
        打印原始数据。

        :return: 无返回值
        """
        print("========== 原始数据 ==========")
        print(self.df)

    def show_missing_info(self):
        """
        查看每一列缺失值数量。

        :return: 无返回值
        """
        print("\n========== 缺失值统计 ==========")
        print(self.df.isnull().sum())

    def handle_outliers(self):
        """
        调用数据预处理类处理异常值。

        :return: 无返回值
        """
        self.df = self.preprocessor.handle_outliers(self.df)

        print("\n========== 异常值处理后数据 ==========")
        print(self.df)

    def prepare_features_and_label(self):
        """
        准备特征 X 和标签 y。

        :return: 无返回值
        """
        feature_columns = self.numeric_features + self.categorical_features

        self.X = self.df[feature_columns]
        self.y = self.df[self.label_column]

        self.y_encoded = self.preprocessor.encode_label(self.y)
        self.preprocessor.show_label_mapping()

    def split_data(self):
        """
        划分训练集和测试集。

        :return: 无返回值
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y_encoded
        )

        print("\n========== 数据集划分结果 ==========")
        print("训练集特征形状：", self.X_train.shape)
        print("测试集特征形状：", self.X_test.shape)
        print("训练集标签形状：", self.y_train.shape)
        print("测试集标签形状：", self.y_test.shape)

    def preprocess_data(self):
        """
        对训练集和测试集进行预处理。

        训练集：
        fit_transform

        测试集：
        transform

        :return: 无返回值
        """
        self.X_train_processed = self.preprocessor.fit_transform_train(
            self.X_train
        )

        self.X_test_processed = self.preprocessor.transform_test(
            self.X_test
        )

        print("\n========== 预处理后数据形状 ==========")
        print("训练集处理后形状：", self.X_train_processed.shape)
        print("测试集处理后形状：", self.X_test_processed.shape)

    def show_processed_train_data(self):
        """
        展示预处理后的训练集数据。

        :return: 无返回值
        """
        feature_names = self.preprocessor.get_processed_feature_names()

        processed_df = pd.DataFrame(
            self.X_train_processed,
            columns=feature_names
        )

        print("\n========== 预处理后的训练集示例 ==========")
        print(processed_df.head())

    def train_model(self):
        """
        训练逻辑回归模型。

        :return: 无返回值
        """
        self.model.fit(self.X_train_processed, self.y_train)
        print("\n模型训练完成。")

    def predict_test_data(self):
        """
        使用测试集进行预测。

        :return: 无返回值
        """
        self.y_pred = self.model.predict(self.X_test_processed)

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
        print(classification_report(self.y_test, self.y_pred))

    def run(self):
        """
        运行完整数据预处理和建模流程。

        :return: 无返回值
        """
        print("========== 第 11 天：数据预处理 ==========")

        self.load_data()
        self.show_raw_data()
        self.show_missing_info()

        self.handle_outliers()
        self.prepare_features_and_label()
        self.split_data()

        self.preprocess_data()
        self.show_processed_train_data()

        self.train_model()
        self.predict_test_data()
        self.evaluate_model()


def main():
    """
    主函数。

    :return: 无返回值
    """
    demo = StudentFailurePredictionDemo(
        test_size=0.3,
        random_state=42
    )

    demo.run()


if __name__ == "__main__":
    main()
十二、代码重点解释
1. 构造原始数据
builder = StudentDataBuilder()
self.df = builder.build_raw_data()

这里构造了一个带问题的数据集，包括：

年龄缺失；
年龄异常；
学习时长缺失；
出勤率缺失；
性别缺失；
专业缺失。

这样更接近真实项目数据。

2. 查看缺失值
self.df.isnull().sum()

它会统计每一列有多少缺失值。

例如：

gender             1
age                2
study_hours        2
attendance_rate    1
major              1
failed             0

这一步是数据预处理前必须做的。

3. 处理异常值
df["age"] = df["age"].clip(lower=0, upper=120)

含义是：

年龄小于 0 的改成 0；
年龄大于 120 的改成 120。

当前案例中，年龄 200 会被截断成 120。

真实项目里，你也可以选择删除这条数据，或者设为缺失值再填充。

4. 标签编码
self.y_encoded = self.preprocessor.encode_label(self.y)

原始标签是：

Yes
No

模型不能直接处理字符串标签，所以要转换成数字。

可能映射为：

No -> 0
Yes -> 1

注意：具体映射关系以代码输出为准。

5. 数值字段处理 Pipeline
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

这表示数值字段依次进行：

中位数填充缺失值；
标准化。

数值字段包括：

age
study_hours
attendance_rate
6. 类别字段处理 Pipeline
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

这表示类别字段依次进行：

众数填充缺失值；
One-Hot 编码。

类别字段包括：

gender
major
7. ColumnTransformer
self.preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, self.numeric_features),
        ("cat", categorical_transformer, self.categorical_features)
    ]
)

它的作用是：

不同类型字段使用不同预处理方式。

也就是：

数值字段走数值处理流程；
类别字段走类别处理流程；
最后把处理结果合并成一个模型可用的特征矩阵。
十三、为什么使用 Pipeline 和 ColumnTransformer？

因为真实项目中，数据预处理流程很容易乱。

例如：

数值字段要填充和标准化；
类别字段要填充和独热编码；
训练集和测试集不能混着 fit；
以后预测新数据也要复用同一套规则。

如果全部手写，很容易出错。

使用 Pipeline 和 ColumnTransformer 的好处是：

结构清晰；
减少数据泄露；
方便复用；
方便和模型组合；
更接近真实项目写法。
十四、运行后重点观察什么？

运行代码后，重点观察 5 个地方。

1. 原始数据

观察数据中是否存在：

NaN；
None；
年龄 200；
字符串类别字段。
2. 缺失值统计

重点看：

哪些列有缺失；
每列缺失几个；
标签列是否有缺失。

标签列如果有缺失，通常不能简单填充，需要根据业务判断是否删除样本。

3. 异常值处理结果

观察：

age = 200 是否被处理成 120。
4. 预处理后数据形状

你会发现原始特征只有 5 个：

age
study_hours
attendance_rate
gender
major

但是 One-Hot 之后，特征数量会增加。

例如：

gender_女
gender_男
major_人工智能
major_物联网
major_软件工程

所以预处理后列数会变多。

5. 预处理后的训练集示例

你会看到类似：

age  study_hours  attendance_rate  gender_女  gender_男  major_人工智能 ...

这说明数据已经变成模型可以训练的数字矩阵。

十五、真实项目中的预处理顺序

一个比较标准的流程是：

读取数据
    ↓
查看数据基本信息
    ↓
查看缺失值
    ↓
查看异常值
    ↓
区分特征列和标签列
    ↓
划分训练集和测试集
    ↓
在训练集上 fit 预处理器
    ↓
用同一个预处理器 transform 测试集
    ↓
训练模型
    ↓
评估模型

重点是：

先划分训练集和测试集，再 fit 预处理规则。

不要把全部数据一起标准化、填充、编码后再划分。

原因是：

这样会让测试集的信息提前进入训练过程，造成数据泄露。
十六、哪些模型需要标准化？
1. 通常需要标准化的模型
KNN
逻辑回归
线性回归
SVM
神经网络
KMeans
PCA

这些模型通常对特征尺度敏感。

2. 通常不强制标准化的模型
决策树
随机森林
GBDT
XGBoost
LightGBM

这些树模型主要根据特征阈值分裂，不太依赖特征尺度。

十七、Label Encoding 和 One-Hot Encoding 怎么选？
1. 有顺序关系的类别

例如：

低 / 中 / 高
小学 / 初中 / 高中 / 大学
差 / 一般 / 好 / 优秀

可以考虑 Label Encoding 或自定义顺序编码。

例如：

低 = 0
中 = 1
高 = 2
2. 没有顺序关系的类别

例如：

城市
专业
性别
商品类目
支付方式

更常用 One-Hot Encoding。

例如：

专业 = 物联网 / 软件工程 / 人工智能

这些专业之间没有天然大小关系，不适合直接编码成 0、1、2。

十八、今日练习
练习 1：判断字段类型

下面字段哪些是数值字段，哪些是类别字段？

字段	数值 / 类别
年龄	数值
性别	类别
城市	类别
月收入	数值
是否购买	类别
商品价格	数值
支付方式	类别
学习时长	数值
练习 2：修改异常值处理

当前代码使用：

df["age"] = df["age"].clip(lower=0, upper=120)

你可以改成：

df.loc[df["age"] > 120, "age"] = np.nan

然后让缺失值填充流程处理它。

观察最终结果是否变化。

练习 3：新增一个类别字段

给原始数据新增字段：

grade：年级

例如：

大一
大二
大三
大四

然后把它加入：

self.categorical_features = [
    "gender",
    "major",
    "grade"
]

观察 One-Hot 后特征数量是否增加。

练习 4：改用 MinMaxScaler

把：

StandardScaler()

改成：

MinMaxScaler()

需要先导入：

from sklearn.preprocessing import MinMaxScaler

观察预处理后的数值特征范围。

练习 5：替换模型

把逻辑回归：

self.model = LogisticRegression(max_iter=1000)

替换为随机森林：

from sklearn.ensemble import RandomForestClassifier

self.model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

思考：

随机森林是否必须标准化？
当前预处理流程中标准化是否还保留？
保留是否会影响随机森林使用？
十九、今日学习检查表
检查项	是否掌握
能理解数据预处理的作用	
能查看缺失值数量	
能用均值、中位数、众数填充缺失值	
能理解异常值处理	
能理解 Label Encoding	
能理解 One-Hot Encoding	
能理解 StandardScaler	
能理解 MinMaxScaler	
能区分训练集 fit_transform 和测试集 transform	
能封装一个 DataPreprocessor 类	
二十、今日总结

第 11 天最重要的是这句话：

数据预处理是把原始脏数据转换成模型可以学习的干净数字特征矩阵。

今天你应该掌握：

缺失值要处理；
异常值要处理；
类别字段要编码；
数值字段可标准化或归一化；
训练集 fit，测试集 transform；
不同类型字段要走不同处理流程。

你现在应该能理解下面这段代码的意义：

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

它背后的含义是：

数值字段：填充缺失值 + 标准化；
类别字段：填充缺失值 + 独热编码；
最后合并成模型可用的数据。