第 12 天开始从“把数据洗干净”进入“把数据变得更有用”。今天重点不是换模型，而是学会从原始字段里构造更能表达业务规律的新特征。

已思考 18s
第 12 天：特征工程 Feature Engineering

第 11 天你学习了 数据预处理，重点是把原始数据处理成模型能吃的格式：

缺失值处理
异常值处理
类别变量编码
标准化
归一化

第 12 天进入更关键的一步：特征工程。

你可以记住一句话：

数据预处理是让数据“能用”；
特征工程是让数据“更有用”。

很多真实机器学习项目里，模型效果不好，不一定是模型不够强，而是特征没有表达出真正的业务规律。

一、今日学习目标

今天需要掌握 8 个内容：

什么是特征工程；
特征工程和数据预处理的区别；
什么是特征构造；
什么是特征组合；
什么是时间特征；
什么是分箱特征；
什么是业务特征；
使用电商订单数据构造用户复购预测特征。

今天最终目标是：

从订单明细数据中，构造用户级别特征，然后训练一个用户是否会复购的模型。
二、什么是特征工程？

特征工程就是根据原始数据，构造出更有价值的模型输入特征。

例如原始订单数据是：

用户ID	下单时间	商品金额	商品类目	是否包邮
U001	2026-06-01	120	            服装	是
U001	2026-06-05	300	            数码	否
U002	2026-06-03	50	            食品	是

这些原始字段可以进一步构造成用户特征：

用户历史订单数
用户累计消费金额
用户平均订单金额
用户最近一次购买距今天数
用户是否经常购买包邮商品
用户购买过几个不同类目
用户是否高频购买

这些新特征通常比单条订单信息更适合预测：

用户是否会复购
用户是否会流失
用户是否高价值
用户是否可能购买某类商品
三、特征工程和数据预处理的区别
对比项	数据预处理	                            特征工程
目标	让数据能进入模型	                    让数据更能表达规律
处理对象	缺失值、异常值、类别编码、标准化	构造新字段、组合字段、业务指标
举例	性别转成 One-Hot	                    用户近 30 天订单数
举例	缺失年龄用中位数填充	                用户平均消费金额
举例	收入标准化      	                    最近一次购买距今天数

简单说：

数据预处理解决“数据脏不脏”；
特征工程解决“数据有没有表达出问题本质”。
四、为什么特征工程重要？

假设你要预测学生是否挂科。

原始数据有：

每日学习时长
出勤率
作业提交次数
考试前复习天数

你可以构造新特征：

学习投入指数 = 每日学习时长 × 出勤率
作业完成率 = 作业提交次数 / 总作业次数
是否低出勤 = 出勤率 < 0.6
是否临时抱佛脚 = 平时学习少，但考试前复习天数多

这些特征比单独的原始字段更有业务意义。

模型不是人，它不知道：

出勤率低 + 学习时长短 = 挂科风险高

你通过特征工程把这种规律表达出来，模型就更容易学到。

五、常见特征工程方法
1. 特征选择

特征选择就是从已有字段中选择对任务有用的字段。

例如用户复购预测中，可能有这些字段：

user_id
username
phone
email
order_count
total_amount
last_order_days

其中：

user_id、username、phone、email 通常不能直接作为有效特征；
order_count、total_amount、last_order_days 更有预测意义。

特征选择的目标是：

去掉无用字段；
去掉泄露字段；
去掉噪声字段；
保留对预测任务有帮助的字段。
2. 特征构造

特征构造就是根据原始字段生成新字段。

例如订单数据中：

订单金额
优惠金额

可以构造：

优惠率 = 优惠金额 / 订单金额

再比如：

下单时间

可以构造：

下单月份
下单星期几
是否周末下单
是否晚上下单
3. 特征组合

特征组合就是把多个特征结合起来。

例如：

学习时长 × 出勤率 = 学习投入指数
订单金额 / 订单数量 = 平均订单金额
累计消费金额 / 活跃天数 = 日均消费能力

特征组合的核心是：

让模型更容易看到多个字段之间的关系。
4. 时间特征

时间字段本身不能直接喂给大多数模型，需要拆解。

例如：

order_time = 2026-06-12 20:30:00

可以拆成：

year = 2026
month = 6
day = 12
hour = 20
weekday = 4
is_weekend = 0

在电商中，时间特征非常重要：

最近一次购买距今天数
近 7 天订单数
近 30 天订单数
首次购买距今天数
用户活跃天数
平均购买间隔
5. 分箱特征

分箱就是把连续数值变成区间类别。

例如年龄：

18-25 岁
26-35 岁
36-45 岁
46 岁以上

消费金额：

低消费
中消费
高消费
超高消费

分箱的作用是：

降低异常值影响；
增强业务解释性；
让模型学习非线性规律。

例如：

消费金额从 50 到 100 的变化，意义可能很大；
消费金额从 10000 到 10050 的变化，意义可能不大。
6. 统计聚合特征

统计聚合特征在真实项目中非常常用。

比如订单明细表是订单级别的，但你要预测用户是否复购，就需要把订单聚合成用户级别。

常见聚合特征：

订单数 count
总消费金额 sum
平均订单金额 mean
最大订单金额 max
最小订单金额 min
消费金额标准差 std
不同商品类目数量 nunique
包邮订单比例 mean
最近一次购买时间 max
首次购买时间 min
六、特征工程中的重要风险：特征泄露

特征泄露是机器学习项目中非常严重的问题。

特征泄露指的是：

模型训练时使用了预测时本不应该知道的信息。

例如你要预测：

用户未来 30 天是否会复购

但是你构造特征时用了：

未来 30 天订单数
未来 30 天消费金额
未来是否下单

这就是特征泄露。

因为真实预测时，你不可能提前知道未来订单。

所以要记住：

特征只能来自预测时间点之前的信息；
标签可以来自预测时间点之后的结果。
七、今日案例：电商用户复购预测

今天构造一个模拟电商订单数据集。

任务是：

根据用户历史订单行为，预测用户未来是否会复购。

原始订单字段：

user_id：用户ID
order_id：订单ID
order_date：下单日期
amount：订单金额
category：商品类目
is_free_shipping：是否包邮
discount_amount：优惠金额

我们要构造用户级别特征：

order_count：历史订单数
total_amount：累计消费金额
avg_amount：平均订单金额
max_amount：最大订单金额
min_amount：最小订单金额
category_count：购买过的类目数量
free_shipping_rate：包邮订单比例
avg_discount_rate：平均优惠率
last_order_days：最近一次购买距观察日期的天数
first_order_days：首次购买距观察日期的天数
active_days：用户活跃跨度
avg_order_interval：平均下单间隔
weekend_order_rate：周末下单比例
favorite_category：最常购买类目
amount_level：消费能力分箱
will_repurchase：是否会复购
八、今日完整代码

新建文件：

day12_feature_engineering.py

代码如下：

"""
第 12 天：特征工程 Feature Engineering。

本案例使用模拟电商订单数据，演示如何从订单明细表构造用户级别特征，
并训练一个用户是否会复购的分类模型。

核心目标：
1. 理解特征工程的作用
2. 理解订单级数据如何聚合为用户级数据
3. 构造统计特征、时间特征、比例特征、分箱特征
4. 避免特征泄露
5. 使用构造后的特征训练模型
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class EcommerceOrderDataBuilder:
    """
    电商订单数据构造类。

    :param user_count: 用户数量
    :param random_state: 随机种子，保证每次生成数据一致
    :return: 无返回值
    """

    def __init__(self, user_count: int = 80, random_state: int = 42):
        """
        初始化电商订单数据构造类。

        :param user_count: 模拟用户数量
        :param random_state: 随机种子
        :return: 无返回值
        """
        self.user_count = user_count
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

        self.categories = ["服装", "数码", "食品", "家居", "美妆"]
        self.order_df = None

    def build_order_data(self) -> pd.DataFrame:
        """
        构造订单明细数据。

        每个用户会有 1 到 8 条历史订单。
        每条订单包含订单日期、金额、类目、是否包邮、优惠金额等字段。

        :return: 订单明细 DataFrame
        """
        rows = []
        order_index = 1

        start_date = pd.Timestamp("2026-01-01")
        end_date = pd.Timestamp("2026-06-01")
        total_days = (end_date - start_date).days

        for user_index in range(1, self.user_count + 1):
            user_id = f"U{user_index:03d}"

            order_count = self.rng.integers(1, 9)

            for _ in range(order_count):
                random_day = int(self.rng.integers(0, total_days))
                order_date = start_date + pd.Timedelta(days=random_day)

                amount = round(float(self.rng.uniform(30, 800)), 2)
                category = self.rng.choice(self.categories)
                is_free_shipping = int(self.rng.choice([0, 1], p=[0.35, 0.65]))

                discount_rate = float(self.rng.uniform(0, 0.35))
                discount_amount = round(amount * discount_rate, 2)

                rows.append({
                    "user_id": user_id,
                    "order_id": f"O{order_index:05d}",
                    "order_date": order_date,
                    "amount": amount,
                    "category": category,
                    "is_free_shipping": is_free_shipping,
                    "discount_amount": discount_amount
                })

                order_index += 1

        self.order_df = pd.DataFrame(rows)
        return self.order_df


class EcommerceFeatureEngineer:
    """
    电商特征工程类。

    :param observation_date: 观察日期，只能使用该日期之前的数据构造特征
    :return: 无返回值
    """

    def __init__(self, observation_date: str = "2026-06-01"):
        """
        初始化电商特征工程类。

        :param observation_date: 观察日期，例如 2026-06-01
        :return: 无返回值
        """
        self.observation_date = pd.Timestamp(observation_date)
        self.feature_df = None

    def add_order_level_features(self, order_df: pd.DataFrame) -> pd.DataFrame:
        """
        构造订单级别辅助特征。

        新增字段：
        1. discount_rate: 优惠率
        2. weekday: 下单星期几
        3. is_weekend: 是否周末下单

        :param order_df: 原始订单明细 DataFrame
        :return: 增加订单级特征后的 DataFrame
        """
        df = order_df.copy()

        df["order_date"] = pd.to_datetime(df["order_date"])

        df["discount_rate"] = df["discount_amount"] / df["amount"]
        df["discount_rate"] = df["discount_rate"].replace([np.inf, -np.inf], 0)
        df["discount_rate"] = df["discount_rate"].fillna(0)

        df["weekday"] = df["order_date"].dt.weekday

        # weekday 中，5 表示星期六，6 表示星期日
        df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

        return df

    def build_user_stat_features(self, order_df: pd.DataFrame) -> pd.DataFrame:
        """
        构造用户统计聚合特征。

        从订单级别聚合到用户级别。

        :param order_df: 增加订单级特征后的订单数据
        :return: 用户级统计特征 DataFrame
        """
        user_features = order_df.groupby("user_id").agg(
            order_count=("order_id", "count"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
            max_amount=("amount", "max"),
            min_amount=("amount", "min"),
            amount_std=("amount", "std"),
            category_count=("category", "nunique"),
            free_shipping_rate=("is_free_shipping", "mean"),
            avg_discount_rate=("discount_rate", "mean"),
            weekend_order_rate=("is_weekend", "mean"),
            first_order_date=("order_date", "min"),
            last_order_date=("order_date", "max")
        ).reset_index()

        # 如果用户只有 1 个订单，amount_std 会是 NaN，这里填充为 0
        user_features["amount_std"] = user_features["amount_std"].fillna(0)

        return user_features

    def build_user_time_features(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """
        构造用户时间相关特征。

        新增字段：
        1. last_order_days: 最近一次购买距观察日期的天数
        2. first_order_days: 首次购买距观察日期的天数
        3. active_days: 用户历史活跃跨度
        4. avg_order_interval: 平均下单间隔

        :param user_features: 用户统计特征 DataFrame
        :return: 增加时间特征后的用户特征 DataFrame
        """
        df = user_features.copy()

        df["last_order_days"] = (
            self.observation_date - df["last_order_date"]
        ).dt.days

        df["first_order_days"] = (
            self.observation_date - df["first_order_date"]
        ).dt.days

        df["active_days"] = (
            df["last_order_date"] - df["first_order_date"]
        ).dt.days

        df["avg_order_interval"] = np.where(
            df["order_count"] > 1,
            df["active_days"] / (df["order_count"] - 1),
            df["first_order_days"]
        )

        return df

    def build_favorite_category_feature(self, order_df: pd.DataFrame) -> pd.DataFrame:
        """
        构造用户最常购买类目特征。

        :param order_df: 订单明细 DataFrame
        :return: user_id 和 favorite_category DataFrame
        """
        category_count_df = (
            order_df
            .groupby(["user_id", "category"])
            .size()
            .reset_index(name="category_order_count")
        )

        category_count_df = category_count_df.sort_values(
            by=["user_id", "category_order_count"],
            ascending=[True, False]
        )

        favorite_category_df = category_count_df.drop_duplicates(
            subset=["user_id"],
            keep="first"
        )

        favorite_category_df = favorite_category_df.rename(
            columns={"category": "favorite_category"}
        )

        return favorite_category_df[["user_id", "favorite_category"]]

    def build_amount_level_feature(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """
        构造消费能力分箱特征。

        根据累计消费金额 total_amount 划分：
        1. low
        2. medium
        3. high
        4. very_high

        :param user_features: 用户特征 DataFrame
        :return: 增加 amount_level 字段后的 DataFrame
        """
        df = user_features.copy()

        df["amount_level"] = pd.cut(
            df["total_amount"],
            bins=[0, 800, 1800, 3200, np.inf],
            labels=["low", "medium", "high", "very_high"],
            include_lowest=True
        )

        df["amount_level"] = df["amount_level"].astype(str)

        return df

    def build_label(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """
        构造模拟标签 will_repurchase。

        说明：
        真实项目中，标签应该来自观察日期之后的一段时间，
        例如观察日期后 30 天内用户是否再次下单。

        当前案例为了教学演示，使用历史特征构造一个模拟标签规则：
        1. 订单数较多
        2. 最近购买较近
        3. 累计消费较高
        4. 包邮比例较高

        :param user_features: 用户特征 DataFrame
        :return: 增加标签列后的 DataFrame
        """
        df = user_features.copy()

        score = (
            0.45 * df["order_count"]
            + 0.002 * df["total_amount"]
            - 0.025 * df["last_order_days"]
            + 1.2 * df["free_shipping_rate"]
            + 0.4 * df["category_count"]
        )

        threshold = score.median()

        df["will_repurchase"] = (score >= threshold).astype(int)

        return df

    def build_features(self, order_df: pd.DataFrame) -> pd.DataFrame:
        """
        运行完整特征工程流程。

        :param order_df: 原始订单明细 DataFrame
        :return: 用户级特征 DataFrame
        """
        order_df = self.add_order_level_features(order_df)

        user_features = self.build_user_stat_features(order_df)
        user_features = self.build_user_time_features(user_features)

        favorite_category_df = self.build_favorite_category_feature(order_df)

        user_features = user_features.merge(
            favorite_category_df,
            on="user_id",
            how="left"
        )

        user_features = self.build_amount_level_feature(user_features)
        user_features = self.build_label(user_features)

        # 日期字段已经转成时间差特征，建模时不再直接使用原始日期
        user_features = user_features.drop(
            columns=["first_order_date", "last_order_date"]
        )

        self.feature_df = user_features

        return self.feature_df


class RepurchasePredictionDemo:
    """
    用户复购预测演示类。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.25, random_state: int = 42):
        """
        初始化用户复购预测演示类。

        :param test_size: 测试集比例
        :param random_state: 随机种子
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state

        self.order_df = None
        self.feature_df = None

        self.label_column = "will_repurchase"

        self.drop_columns = [
            "user_id",
            self.label_column
        ]

        self.numeric_features = [
            "order_count",
            "total_amount",
            "avg_amount",
            "max_amount",
            "min_amount",
            "amount_std",
            "category_count",
            "free_shipping_rate",
            "avg_discount_rate",
            "weekend_order_rate",
            "last_order_days",
            "first_order_days",
            "active_days",
            "avg_order_interval"
        ]

        self.categorical_features = [
            "favorite_category",
            "amount_level"
        ]

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.model_pipeline = None
        self.y_pred = None

    def build_data(self):
        """
        构造原始订单数据和用户特征数据。

        :return: 无返回值
        """
        data_builder = EcommerceOrderDataBuilder(
            user_count=80,
            random_state=self.random_state
        )

        self.order_df = data_builder.build_order_data()

        feature_engineer = EcommerceFeatureEngineer(
            observation_date="2026-06-01"
        )

        self.feature_df = feature_engineer.build_features(self.order_df)

    def show_raw_order_data(self):
        """
        展示原始订单明细数据。

        :return: 无返回值
        """
        print("========== 原始订单数据示例 ==========")
        print(self.order_df.head(10))

        print("\n订单数据形状：", self.order_df.shape)

    def show_feature_data(self):
        """
        展示构造后的用户级特征数据。

        :return: 无返回值
        """
        print("\n========== 用户级特征数据示例 ==========")
        print(self.feature_df.head(10))

        print("\n用户特征数据形状：", self.feature_df.shape)

        print("\n========== 标签分布 ==========")
        print(self.feature_df[self.label_column].value_counts())

    def prepare_features_and_label(self):
        """
        准备模型训练所需的 X 和 y。

        :return: 无返回值
        """
        feature_columns = self.numeric_features + self.categorical_features

        self.X = self.feature_df[feature_columns]
        self.y = self.feature_df[self.label_column]

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

    def build_model_pipeline(self):
        """
        构建包含预处理和模型训练的 Pipeline。

        数值特征：
        1. 缺失值用中位数填充
        2. 使用 StandardScaler 标准化

        类别特征：
        1. 缺失值用众数填充
        2. 使用 OneHotEncoder 编码

        模型：
        RandomForestClassifier

        :return: 无返回值
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

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features)
            ]
        )

        classifier = RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            random_state=self.random_state
        )

        self.model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ]
        )

    def train_model(self):
        """
        训练模型。

        :return: 无返回值
        """
        self.model_pipeline.fit(self.X_train, self.y_train)
        print("\n模型训练完成。")

    def predict_test_data(self):
        """
        使用测试集进行预测。

        :return: 无返回值
        """
        self.y_pred = self.model_pipeline.predict(self.X_test)

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

    def show_feature_importance(self):
        """
        输出随机森林特征重要性。

        注意：
        因为类别特征经过 One-Hot 展开，所以需要从预处理器中取出展开后的特征名。

        :return: 无返回值
        """
        preprocessor = self.model_pipeline.named_steps["preprocessor"]
        classifier = self.model_pipeline.named_steps["classifier"]

        numeric_names = self.numeric_features

        onehot_encoder = (
            preprocessor
            .named_transformers_["cat"]
            .named_steps["onehot"]
        )

        categorical_names = onehot_encoder.get_feature_names_out(
            self.categorical_features
        ).tolist()

        all_feature_names = numeric_names + categorical_names

        importance_df = pd.DataFrame({
            "feature": all_feature_names,
            "importance": classifier.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="importance",
            ascending=False
        )

        print("\n========== 特征重要性 Top 15 ==========")
        print(importance_df.head(15))

    def run(self):
        """
        运行完整特征工程和建模流程。

        :return: 无返回值
        """
        print("========== 第 12 天：特征工程 Feature Engineering ==========")

        self.build_data()
        self.show_raw_order_data()
        self.show_feature_data()

        self.prepare_features_and_label()
        self.split_data()

        self.build_model_pipeline()
        self.train_model()
        self.predict_test_data()
        self.evaluate_model()
        self.show_feature_importance()


def main():
    """
    主函数。

    :return: 无返回值
    """
    demo = RepurchasePredictionDemo(
        test_size=0.25,
        random_state=42
    )

    demo.run()


if __name__ == "__main__":
    main()
九、代码重点解释
1. 原始订单数据是订单级别

原始数据类似：

user_id
order_id
order_date
amount
category
is_free_shipping
discount_amount

每一行是一笔订单。

但是我们的预测目标是：

用户是否会复购

所以建模粒度应该是：

用户级别

这就是为什么要做聚合特征。

2. 构造订单级辅助特征
df["discount_rate"] = df["discount_amount"] / df["amount"]
df["weekday"] = df["order_date"].dt.weekday
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

这些是从单条订单中构造出来的辅助特征。

含义是：

discount_rate：优惠率
weekday：星期几下单
is_weekend：是否周末下单
3. 从订单级聚合到用户级

核心代码：

user_features = order_df.groupby("user_id").agg(
    order_count=("order_id", "count"),
    total_amount=("amount", "sum"),
    avg_amount=("amount", "mean"),
    max_amount=("amount", "max"),
    min_amount=("amount", "min"),
    category_count=("category", "nunique"),
    free_shipping_rate=("is_free_shipping", "mean")
).reset_index()

这一步非常重要。

它的含义是：

把一个用户的多条订单，汇总成这个用户的一行特征。

例如用户 U001 有 5 条订单，聚合后变成：

用户 U001：
订单数 = 5
累计消费 = 2300
平均订单金额 = 460
购买类目数 = 3
包邮订单比例 = 0.8
4. 构造时间特征
df["last_order_days"] = (
    self.observation_date - df["last_order_date"]
).dt.days

含义是：

最近一次购买距观察日期过去了多少天。

这个特征很重要。

一般来说：

最近刚买过的用户，可能更活跃；
很久没买过的用户，可能已经沉睡。

再比如：

df["active_days"] = (
    df["last_order_date"] - df["first_order_date"]
).dt.days

表示用户历史活跃跨度。

5. 构造平均下单间隔
df["avg_order_interval"] = np.where(
    df["order_count"] > 1,
    df["active_days"] / (df["order_count"] - 1),
    df["first_order_days"]
)

含义是：

如果用户有多个订单，就计算平均多久下一次单；
如果用户只有一个订单，就用首次购买距观察日期天数作为近似。

这个特征能表达用户购买频率。

6. 构造最常购买类目
favorite_category_df = category_count_df.drop_duplicates(
    subset=["user_id"],
    keep="first"
)

作用是找出每个用户最常购买的商品类目。

例如：

U001 最常购买：服装
U002 最常购买：食品
U003 最常购买：数码

这是一个类别特征，后续要做 One-Hot Encoding。

7. 构造分箱特征
df["amount_level"] = pd.cut(
    df["total_amount"],
    bins=[0, 800, 1800, 3200, np.inf],
    labels=["low", "medium", "high", "very_high"],
    include_lowest=True
)

这一步把累计消费金额分成：

low
medium
high
very_high

分箱后的字段更容易解释：

低消费用户
中消费用户
高消费用户
超高消费用户
十、今日构造的特征清单
特征名	含义	类型
order_count	用户历史订单数	统计特征
total_amount	用户累计消费金额	统计特征
avg_amount	用户平均订单金额	统计特征
max_amount	用户最大订单金额	统计特征
min_amount	用户最小订单金额	统计特征
amount_std	用户订单金额波动	统计特征
category_count	用户购买过的类目数量	统计特征
free_shipping_rate	包邮订单比例	比例特征
avg_discount_rate	平均优惠率	比例特征
weekend_order_rate	周末下单比例	时间比例特征
last_order_days	最近一次购买距观察日期天数	时间特征
first_order_days	首次购买距观察日期天数	时间特征
active_days	用户活跃跨度	时间特征
avg_order_interval	平均下单间隔	时间特征
favorite_category	最常购买类目	类别特征
amount_level	消费能力分箱	分箱特征
十一、运行后重点观察什么？
1. 原始订单数据形状

你会看到类似：

订单数据形状：(几百行, 7列)

表示：

一行是一条订单；
一个用户可能有多条订单。
2. 用户特征数据形状

你会看到类似：

用户特征数据形状：(80行, 多列)

表示：

一行是一个用户；
每个用户有一组聚合后的特征。

这就是从订单级别到用户级别的转换。

3. 标签分布
will_repurchase
0    40
1    40

这里是模拟标签，所以大致均衡。

真实项目中，复购标签可能不均衡，例如：

未复购用户很多；
复购用户较少。

这时第 6 天学过的 Precision、Recall、F1-score 就很重要。

4. 特征重要性

随机森林会输出：

特征重要性 Top 15

你要重点观察：

order_count 是否重要；
last_order_days 是否重要；
total_amount 是否重要；
free_shipping_rate 是否重要；
category_count 是否重要。

这些特征的业务意义很强。

十二、真实项目中特征工程怎么做？

真实项目一般会按下面流程：

明确预测目标
    ↓
确定样本粒度
    ↓
确定观察窗口
    ↓
确定标签窗口
    ↓
构造原始特征
    ↓
构造统计特征
    ↓
构造时间特征
    ↓
构造比例特征
    ↓
构造分箱特征
    ↓
避免特征泄露
    ↓
训练模型并验证特征效果
十三、什么是样本粒度？

样本粒度就是一行数据代表什么。

例如：

一行是一个用户
一行是一个订单
一行是一个商品
一行是一个学生
一行是一篇文章
一行是一张图片

今天的任务是用户复购预测，所以样本粒度是：

一行 = 一个用户

如果样本粒度搞错，模型就会非常混乱。

十四、什么是观察窗口和标签窗口？

这是时间类机器学习任务中非常重要的概念。

假设今天是：

2026-06-01

你要预测：

未来 30 天用户是否复购

那么：

观察窗口：2026-01-01 到 2026-06-01
标签窗口：2026-06-02 到 2026-07-01

特征只能用观察窗口的数据：

历史订单数
历史消费金额
最近一次购买时间

标签来自标签窗口：

未来 30 天是否下单

绝对不能用标签窗口的数据构造特征。

否则就是特征泄露。

十五、常见业务特征举例
1. 电商用户复购预测
历史订单数
近 7 天订单数
近 30 天订单数
累计消费金额
平均订单金额
最大订单金额
最近一次购买距今天数
购买类目数量
优惠券使用率
包邮订单比例
退款次数
加购次数
收藏次数
浏览次数
2. 学生挂科预测
出勤率
作业提交次数
作业完成率
课堂测验平均分
学习时长
迟到次数
缺勤次数
实验完成次数
上机训练次数
期中成绩
最近一次测验距今天数
3. 设备故障预测
运行时长
最近一次维修距今天数
历史故障次数
平均温度
最大温度
温度波动
电压波动
异常报警次数
近 7 天报警次数
近 30 天报警次数
4. 文本分类任务
文本长度
词数量
平均词长
关键词出现次数
情感词数量
问号数量
感叹号数量
TF-IDF 特征
关键词类别特征
十六、特征工程中的几个原则
1. 特征要和目标有关

不是字段越多越好。

你要问：

这个特征和我要预测的目标有没有关系？

例如预测复购：

order_count 有关系；
last_order_days 有关系；
用户手机号后四位通常没关系。
2. 特征不能泄露未来

预测未来时，只能用过去的数据。

例如：

预测未来是否复购，不能用未来订单数作为特征。
3. 特征要能在预测时拿到

有些字段训练时有，但真实预测时没有。

这种字段不能用。

例如：

用户最终是否退款
客服最终处理结果
未来 30 天消费金额

如果预测时拿不到，就不能作为特征。

4. 先做简单特征，再做复杂特征

不要一开始就设计特别复杂的特征。

推荐顺序：

基础统计特征
    ↓
时间特征
    ↓
比例特征
    ↓
分箱特征
    ↓
组合特征
    ↓
更复杂的业务特征
十七、今日练习
练习 1：解释特征工程

请用自己的话解释：

数据预处理和特征工程有什么区别？

参考方向：

数据预处理是处理缺失值、异常值、编码和标准化；
特征工程是根据业务和原始字段构造更有预测价值的新特征。
练习 2：新增一个用户特征

在代码中新增一个特征：

avg_discount_amount：平均优惠金额

提示：

avg_discount_amount=("discount_amount", "mean")

加入到 build_user_stat_features() 里面。

练习 3：新增近 30 天订单数

构造一个特征：

last_30_days_order_count：最近 30 天订单数

提示思路：

recent_order_df = order_df[
    order_df["order_date"] >= self.observation_date - pd.Timedelta(days=30)
]

然后按 user_id 聚合订单数，再 merge 回用户特征表。

练习 4：新增高价值用户标记

根据累计消费金额构造：

is_high_value_user

规则：

total_amount >= 3000，记为 1；
否则记为 0。
练习 5：检查特征泄露

判断下面字段能不能作为预测未来 30 天复购的特征：

字段	能否作为特征	原因
历史订单数	能	来自过去
未来 30 天订单数	不能	泄露未来
最近一次购买距今天数	能	来自过去
未来 30 天消费金额	不能	泄露未来
历史包邮订单比例	能	来自过去
是否在未来 30 天下单	不能	这是标签，不是特征
十八、今日学习检查表
检查项	是否掌握
能理解特征工程的作用	
能区分数据预处理和特征工程	
能理解样本粒度	
能理解观察窗口和标签窗口	
能构造统计聚合特征	
能构造时间特征	
能构造比例特征	
能构造分箱特征	
能理解特征泄露	
能从订单明细构造用户特征	
十九、今日总结

第 12 天最重要的是这句话：

特征工程就是把原始数据转换成更能表达业务规律的模型输入。

今天你应该掌握：

订单级数据可以聚合成用户级特征；
时间字段可以转成最近一次购买距今天数；
消费金额可以构造累计值、平均值、最大值、分箱；
类别字段可以构造最常购买类目；
比例特征可以表达用户偏好；
不能使用未来信息构造特征。

今天你真正要建立的意识是：

模型不是万能的，模型只能学习你给它的特征。