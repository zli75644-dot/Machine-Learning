第 15 天从“有标签的监督学习”切换到“没有标签的无监督学习”。今天重点是聚类：模型不再预测标准答案，而是自己发现数据中的群体结构。

已思考 12s
第 15 天：无监督学习与聚类 Clustering

前 14 天你主要学习的是监督学习。

监督学习的特点是：

有特征 X；
也有标准答案 y；
模型通过 X 和 y 学习规律。

例如：

房价预测：X = 面积、楼层、房间数，y = 房价
肿瘤分类：X = 细胞特征，y = 良性 / 恶性
鸢尾花分类：X = 花萼花瓣特征，y = 花的类别

第 15 天开始学习 无监督学习。

无监督学习的特点是：

只有特征 X；
没有标准答案 y；
模型需要自己从数据中发现结构。
一、今日学习目标

今天需要掌握 8 个内容：

什么是无监督学习；
聚类和分类有什么区别；
什么是 KMeans 聚类；
KMeans 是如何一步一步分组的；
什么是肘部法则 Elbow Method；
什么是轮廓系数 Silhouette Score；
什么是 DBSCAN 聚类；
用聚类方法完成用户分层。
二、什么是无监督学习？

无监督学习就是没有标签答案的学习。

例如你有一批用户数据：

用户	年消费金额	下单次数	平均客单价	退款率
U001	500	             3	    166     	0.02
U002	8000	        40	    200	        0.01
U003	1200	        5	    240	        0.20
U004	15000	        60	    250	        0.03

你没有告诉模型：

谁是高价值用户；
谁是普通用户；
谁是风险用户。

但你希望模型根据用户行为自动分组。

这就是聚类任务。

三、聚类和分类的区别
对比项	分类 Classification	聚类 Clustering
学习类型	监督学习	无监督学习
是否有标签 y	有	        没有
目标	学会预测已知类别	自动发现数据分组
例子	判断肿瘤良性/恶性	用户自动分层
输出含义	类别有明确业务含义	类别需要后续解释
常用算法	逻辑回归、KNN、随机森林	KMeans、DBSCAN

一句话总结：

分类是模型学习已有答案；
聚类是模型自己发现分组。
四、聚类能用来做什么？

聚类在真实项目中很常见。

例如：

用户分层；
商品分组；
客户画像；
异常用户发现；
相似文章聚合；
相似图片聚合；
城市消费水平分组；
学生学习行为分组；
设备运行状态分组。

比如用户分层中，聚类可能把用户分成：

高消费高频用户；
低消费低频用户；
高退款风险用户；
优惠敏感型用户；
沉睡用户。

注意：

聚类算法只会给出第 0 类、第 1 类、第 2 类；
这些类别到底代表什么，需要我们结合业务解释。
五、KMeans 聚类是什么？

KMeans 是最经典的聚类算法之一。

它的目标是：

把数据分成 K 个簇，让同一簇内的样本尽量相似，不同簇之间尽量不同。

其中：

K = 你希望分成几类。

例如：

K = 3

表示希望模型把用户分成 3 组。

六、KMeans 的核心思想

KMeans 的过程可以理解为：

第 1 步：随机选 K 个中心点；
第 2 步：每个样本找到离自己最近的中心点；
第 3 步：把样本分配给最近的中心点；
第 4 步：每个簇重新计算中心点；
第 5 步：重复第 2 到第 4 步；
第 6 步：直到分组基本稳定。

通俗理解：

先随便定几个小组长；
每个用户找离自己最近的小组长；
每个小组根据成员重新选中心；
不断调整，直到小组基本不变。
七、KMeans 的目标函数

KMeans 希望每个样本离自己所在簇中心越近越好。

这里你先这样理解：

x_i：第 i 个样本；
μ：簇中心；
c_i：第 i 个样本所属的簇；
J：所有样本到自己簇中心的距离平方和。

KMeans 的目标就是：

让 J 尽可能小。

也就是让同一组内的样本尽量靠近。

八、KMeans 为什么需要标准化？

KMeans 和 KNN 一样，都依赖距离。

如果特征尺度差别很大，距离会被数值大的特征主导。

例如用户数据：

年消费金额：100 到 50000
退款率：0 到 1
下单次数：1 到 100

如果不标准化，年消费金额的数值范围最大，会主导距离计算。

所以 KMeans 通常需要：

StandardScaler()

让不同特征处在接近的尺度上。

九、K 值怎么选？

KMeans 必须提前指定 K。

但是很多时候你不知道应该分几类。

常用方法有两个：

肘部法则 Elbow Method；
轮廓系数 Silhouette Score。
1. 肘部法则 Elbow Method

KMeans 中有一个指标叫：

inertia

它表示：

样本到自己所在簇中心的距离平方和。

K 越大，分得越细，inertia 通常越小。

但是 K 不能无限大。

如果 K 等于样本数量，每个样本自己一类，inertia 会非常小，但没有业务意义。

肘部法则就是观察：

当 K 增加时，inertia 下降速度什么时候明显变慢。

下降速度明显变慢的位置，就像手肘一样。

这个 K 可以作为参考。

2. 轮廓系数 Silhouette Score

轮廓系数衡量聚类效果。

它大致关注两件事：

同一簇内是否足够紧密；
不同簇之间是否分得足够开。

轮廓系数范围一般是：

-1 到 1

越接近 1，说明聚类效果越好。

可以这样理解：

轮廓系数	含义
接近 1	聚类效果较好
接近 0	簇之间边界不明显
小于 0	可能分错了
十、DBSCAN 聚类是什么？

DBSCAN 也是一种常见聚类算法。

它和 KMeans 不一样。

KMeans 需要提前指定：

K = 分成几类

DBSCAN 不需要提前指定 K。

DBSCAN 主要根据：

样本密度

来聚类。

如果一片区域样本很密集，它就认为这些样本是一类。

如果某些点周围很孤立，它可能认为这些点是噪声点。

十一、DBSCAN 的两个重要参数
1. eps
eps=0.5

表示邻域半径。

可以理解为：

在多大范围内找邻居。

eps 太小：

很多点可能被认为是噪声。

eps 太大：

很多簇可能被合并成一个大簇。
2. min_samples
min_samples=5

表示一个点周围至少有多少个点，才算密集区域。

可以理解为：

至少多少个邻居才能形成一个核心区域。
十二、KMeans 和 DBSCAN 的区别
对比项	            KMeans      	DBSCAN
是否要指定类别数	需要指定 K	    不需要指定 K
核心思想	        按距离找中心点  按密度找簇
是否能发现噪声点	不擅长      	擅长
簇形状	            更适合球形簇	可发现不规则形状
对标准化	        敏感	        敏感
常见参数	        n_clusters  	eps、min_samples

入门阶段你先记住：

KMeans 适合分成几个相对规则的群体；
DBSCAN 适合发现密集区域和异常点。
十三、今日案例：用户分层聚类

今天构造一批模拟用户数据。

字段包括：

annual_spend：年消费金额
order_count：年下单次数
avg_order_amount：平均客单价
discount_rate：平均优惠率
return_rate：退款率

目标是：

不用标签 y，让模型自动把用户分成不同群体。
十四、今日完整代码

新建文件：

day15_clustering.py

代码如下：

"""
第 15 天：无监督学习与聚类 Clustering。

本案例使用模拟用户消费数据，演示无监督学习中的聚类任务。

核心目标：
1. 理解分类和聚类的区别
2. 理解 KMeans 聚类
3. 理解肘部法则 Elbow Method
4. 理解轮廓系数 Silhouette Score
5. 理解 DBSCAN 聚类
6. 用聚类做用户分层
7. 对聚类结果进行业务解释
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


class UserDataBuilder:
    """
    用户消费数据构造类。

    :param random_state: 随机种子，保证每次生成数据一致
    :return: 无返回值
    """

    def __init__(self, random_state: int = 42):
        """
        初始化用户消费数据构造类。

        :param random_state: 随机种子
        :return: 无返回值
        """
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        self.df = None

    def build_user_data(self) -> pd.DataFrame:
        """
        构造模拟用户数据。

        为了方便观察聚类效果，这里模拟 4 类用户：
        1. 普通低频用户
        2. 高价值高频用户
        3. 优惠敏感用户
        4. 高退款风险用户

        注意：
        真实聚类任务中没有这些标签。
        这里的模拟只是为了让数据本身具有分组结构。

        :return: 用户数据 DataFrame
        """
        rows = []
        user_index = 1

        group_configs = [
            {
                "group_name": "普通低频用户",
                "count": 60,
                "annual_spend_mean": 1200,
                "annual_spend_std": 300,
                "order_count_mean": 5,
                "order_count_std": 2,
                "discount_rate_mean": 0.12,
                "return_rate_mean": 0.03
            },
            {
                "group_name": "高价值高频用户",
                "count": 60,
                "annual_spend_mean": 12000,
                "annual_spend_std": 2500,
                "order_count_mean": 45,
                "order_count_std": 8,
                "discount_rate_mean": 0.08,
                "return_rate_mean": 0.02
            },
            {
                "group_name": "优惠敏感用户",
                "count": 60,
                "annual_spend_mean": 4500,
                "annual_spend_std": 1000,
                "order_count_mean": 18,
                "order_count_std": 5,
                "discount_rate_mean": 0.35,
                "return_rate_mean": 0.05
            },
            {
                "group_name": "高退款风险用户",
                "count": 60,
                "annual_spend_mean": 6500,
                "annual_spend_std": 1500,
                "order_count_mean": 22,
                "order_count_std": 6,
                "discount_rate_mean": 0.18,
                "return_rate_mean": 0.28
            }
        ]

        for config in group_configs:
            for _ in range(config["count"]):
                annual_spend = max(
                    100,
                    self.rng.normal(
                        config["annual_spend_mean"],
                        config["annual_spend_std"]
                    )
                )

                order_count = max(
                    1,
                    int(
                        self.rng.normal(
                            config["order_count_mean"],
                            config["order_count_std"]
                        )
                    )
                )

                avg_order_amount = annual_spend / order_count

                discount_rate = np.clip(
                    self.rng.normal(config["discount_rate_mean"], 0.04),
                    0,
                    0.8
                )

                return_rate = np.clip(
                    self.rng.normal(config["return_rate_mean"], 0.035),
                    0,
                    0.8
                )

                rows.append({
                    "user_id": f"U{user_index:04d}",
                    "annual_spend": round(float(annual_spend), 2),
                    "order_count": order_count,
                    "avg_order_amount": round(float(avg_order_amount), 2),
                    "discount_rate": round(float(discount_rate), 4),
                    "return_rate": round(float(return_rate), 4),

                    # 这个字段只是为了教学验证，不参与聚类训练
                    "true_group_for_demo": config["group_name"]
                })

                user_index += 1

        self.df = pd.DataFrame(rows)
        return self.df


class UserClusteringAnalyzer:
    """
    用户聚类分析类。

    :param feature_columns: 用于聚类的特征列
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, feature_columns: list, random_state: int = 42):
        """
        初始化用户聚类分析类。

        :param feature_columns: 用于聚类的特征列
        :param random_state: 随机种子
        :return: 无返回值
        """
        self.feature_columns = feature_columns
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.X = None
        self.X_scaled = None

        self.kmeans_model = None
        self.dbscan_model = None

        self.kmeans_result_df = None
        self.dbscan_result_df = None

        self.elbow_df = None
        self.silhouette_df = None

    def prepare_features(self, df: pd.DataFrame):
        """
        准备用于聚类的特征，并完成标准化。

        聚类算法依赖距离，因此通常需要标准化。

        :param df: 原始用户数据 DataFrame
        :return: 标准化后的特征矩阵
        """
        self.X = df[self.feature_columns]
        self.X_scaled = self.scaler.fit_transform(self.X)

        print("\n========== 聚类特征 ==========")
        print(self.X.head())

        print("\n========== 标准化后特征形状 ==========")
        print(self.X_scaled.shape)

        return self.X_scaled

    def run_kmeans(self, df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
        """
        运行 KMeans 聚类。

        :param df: 原始用户数据 DataFrame
        :param n_clusters: 聚类数量 K
        :return: 增加 cluster 字段后的 DataFrame
        """
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )

        cluster_labels = self.kmeans_model.fit_predict(self.X_scaled)

        result_df = df.copy()
        result_df["kmeans_cluster"] = cluster_labels

        self.kmeans_result_df = result_df

        print(f"\n========== KMeans 聚类完成，K={n_clusters} ==========")
        print(result_df["kmeans_cluster"].value_counts().sort_index())

        return result_df

    def analyze_kmeans_clusters(self):
        """
        分析 KMeans 聚类结果。

        通过每个簇的特征均值，给聚类结果做业务解释。

        :return: 聚类画像 DataFrame
        """
        if self.kmeans_result_df is None:
            raise ValueError("请先运行 run_kmeans()。")

        profile_df = (
            self.kmeans_result_df
            .groupby("kmeans_cluster")[self.feature_columns]
            .mean()
            .reset_index()
        )

        count_df = (
            self.kmeans_result_df
            .groupby("kmeans_cluster")
            .size()
            .reset_index(name="user_count")
        )

        profile_df = profile_df.merge(
            count_df,
            on="kmeans_cluster",
            how="left"
        )

        print("\n========== KMeans 用户分群画像 ==========")
        print(profile_df)

        return profile_df

    def run_elbow_method(self, max_k: int = 10) -> pd.DataFrame:
        """
        使用肘部法则观察不同 K 值下的 inertia。

        :param max_k: 最大 K 值
        :return: K 值和 inertia 的 DataFrame
        """
        results = []

        for k in range(1, max_k + 1):
            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )

            model.fit(self.X_scaled)

            results.append({
                "k": k,
                "inertia": model.inertia_
            })

        self.elbow_df = pd.DataFrame(results)

        print("\n========== 肘部法则结果 ==========")
        print(self.elbow_df)

        return self.elbow_df

    def plot_elbow_curve(self):
        """
        绘制肘部法则曲线。

        :return: 无返回值
        """
        if self.elbow_df is None:
            raise ValueError("请先运行 run_elbow_method()。")

        plt.figure(figsize=(8, 5))
        plt.plot(
            self.elbow_df["k"],
            self.elbow_df["inertia"],
            marker="o"
        )
        plt.xlabel("K")
        plt.ylabel("Inertia")
        plt.title("Elbow Method")
        plt.xticks(self.elbow_df["k"])
        plt.grid(True)
        plt.show()

    def run_silhouette_analysis(self, min_k: int = 2, max_k: int = 10) -> pd.DataFrame:
        """
        计算不同 K 值下的轮廓系数。

        注意：
        轮廓系数至少需要 2 个簇，所以从 K=2 开始。

        :param min_k: 最小 K 值
        :param max_k: 最大 K 值
        :return: K 值和 silhouette_score 的 DataFrame
        """
        results = []

        for k in range(min_k, max_k + 1):
            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )

            labels = model.fit_predict(self.X_scaled)

            score = silhouette_score(self.X_scaled, labels)

            results.append({
                "k": k,
                "silhouette_score": score
            })

        self.silhouette_df = pd.DataFrame(results)

        print("\n========== 轮廓系数结果 ==========")
        print(self.silhouette_df)

        return self.silhouette_df

    def plot_silhouette_curve(self):
        """
        绘制轮廓系数曲线。

        :return: 无返回值
        """
        if self.silhouette_df is None:
            raise ValueError("请先运行 run_silhouette_analysis()。")

        plt.figure(figsize=(8, 5))
        plt.plot(
            self.silhouette_df["k"],
            self.silhouette_df["silhouette_score"],
            marker="o"
        )
        plt.xlabel("K")
        plt.ylabel("Silhouette Score")
        plt.title("K Value vs Silhouette Score")
        plt.xticks(self.silhouette_df["k"])
        plt.grid(True)
        plt.show()

    def run_dbscan(self, df: pd.DataFrame, eps: float = 0.8, min_samples: int = 5) -> pd.DataFrame:
        """
        运行 DBSCAN 聚类。

        DBSCAN 的标签中：
        -1 表示噪声点。

        :param df: 原始用户数据 DataFrame
        :param eps: 邻域半径
        :param min_samples: 形成核心点所需的最小样本数
        :return: 增加 dbscan_cluster 字段后的 DataFrame
        """
        self.dbscan_model = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        labels = self.dbscan_model.fit_predict(self.X_scaled)

        result_df = df.copy()
        result_df["dbscan_cluster"] = labels

        self.dbscan_result_df = result_df

        print(f"\n========== DBSCAN 聚类完成，eps={eps}, min_samples={min_samples} ==========")
        print(result_df["dbscan_cluster"].value_counts().sort_index())

        noise_count = (labels == -1).sum()
        print("噪声点数量：", noise_count)

        return result_df

    def analyze_dbscan_clusters(self):
        """
        分析 DBSCAN 聚类结果。

        :return: DBSCAN 聚类画像 DataFrame
        """
        if self.dbscan_result_df is None:
            raise ValueError("请先运行 run_dbscan()。")

        profile_df = (
            self.dbscan_result_df
            .groupby("dbscan_cluster")[self.feature_columns]
            .mean()
            .reset_index()
        )

        count_df = (
            self.dbscan_result_df
            .groupby("dbscan_cluster")
            .size()
            .reset_index(name="user_count")
        )

        profile_df = profile_df.merge(
            count_df,
            on="dbscan_cluster",
            how="left"
        )

        print("\n========== DBSCAN 用户分群画像 ==========")
        print(profile_df)

        return profile_df

    def visualize_kmeans_result(self):
        """
        使用 annual_spend 和 order_count 两个特征可视化 KMeans 聚类结果。

        注意：
        实际聚类使用了多个特征。
        这里仅用二维图辅助理解。

        :return: 无返回值
        """
        if self.kmeans_result_df is None:
            raise ValueError("请先运行 run_kmeans()。")

        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.kmeans_result_df["annual_spend"],
            self.kmeans_result_df["order_count"],
            c=self.kmeans_result_df["kmeans_cluster"]
        )
        plt.xlabel("Annual Spend")
        plt.ylabel("Order Count")
        plt.title("KMeans User Clustering")
        plt.grid(True)
        plt.show()


class ClusteringWorkflow:
    """
    聚类完整流程类。

    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, random_state: int = 42):
        """
        初始化聚类完整流程类。

        :param random_state: 随机种子
        :return: 无返回值
        """
        self.random_state = random_state

        self.user_df = None

        self.feature_columns = [
            "annual_spend",
            "order_count",
            "avg_order_amount",
            "discount_rate",
            "return_rate"
        ]

        self.analyzer = UserClusteringAnalyzer(
            feature_columns=self.feature_columns,
            random_state=self.random_state
        )

    def run(self):
        """
        运行完整聚类流程。

        :return: 无返回值
        """
        print("========== 第 15 天：无监督学习与聚类 ==========")

        builder = UserDataBuilder(
            random_state=self.random_state
        )

        self.user_df = builder.build_user_data()

        print("========== 原始用户数据示例 ==========")
        print(self.user_df.head(10))

        print("\n========== 原始数据形状 ==========")
        print(self.user_df.shape)

        print("\n========== 教学用真实分组统计，不参与聚类 ==========")
        print(self.user_df["true_group_for_demo"].value_counts())

        self.analyzer.prepare_features(self.user_df)

        self.analyzer.run_elbow_method(max_k=10)
        self.analyzer.plot_elbow_curve()

        self.analyzer.run_silhouette_analysis(min_k=2, max_k=10)
        self.analyzer.plot_silhouette_curve()

        self.analyzer.run_kmeans(
            df=self.user_df,
            n_clusters=4
        )

        self.analyzer.analyze_kmeans_clusters()
        self.analyzer.visualize_kmeans_result()

        self.analyzer.run_dbscan(
            df=self.user_df,
            eps=0.9,
            min_samples=5
        )

        self.analyzer.analyze_dbscan_clusters()


def main():
    """
    主函数。

    :return: 无返回值
    """
    workflow = ClusteringWorkflow(
        random_state=42
    )

    workflow.run()


if __name__ == "__main__":
    main()
十五、代码重点解释
1. 为什么没有 y？

聚类是无监督学习。

所以代码中用于聚类的只有：

self.feature_columns = [
    "annual_spend",
    "order_count",
    "avg_order_amount",
    "discount_rate",
    "return_rate"
]

没有：

y_train
y_test

因为聚类没有标准答案。

代码里的：

true_group_for_demo

只是为了教学演示，让你知道模拟数据原本设计了几种人群。

真实聚类项目中通常没有这个字段。

2. 为什么要标准化？

代码中：

self.X_scaled = self.scaler.fit_transform(self.X)

因为 KMeans 和 DBSCAN 都依赖距离。

如果不标准化：

annual_spend 数值范围很大；
discount_rate 数值范围很小；
距离计算会被 annual_spend 主导。

标准化后，每个特征在距离计算中的影响会更公平。

3. KMeans 聚类
self.kmeans_model = KMeans(
    n_clusters=n_clusters,
    random_state=self.random_state,
    n_init=10
)

这里：

n_clusters=4 表示分成 4 类；
n_init=10 表示用 10 次不同初始化，选择较好结果。

KMeans 会输出每个用户属于哪个簇：

result_df["kmeans_cluster"] = cluster_labels
4. 聚类画像分析
profile_df = (
    self.kmeans_result_df
    .groupby("kmeans_cluster")[self.feature_columns]
    .mean()
    .reset_index()
)

这一步非常重要。

聚类算法只会告诉你：

用户属于第 0 类、第 1 类、第 2 类、第 3 类。

但这些数字没有业务含义。

你需要通过每一类的特征均值解释：

第 0 类：年消费高、下单多、退款低 -> 高价值用户
第 1 类：年消费低、下单少 -> 普通低频用户
第 2 类：优惠率高 -> 优惠敏感用户
第 3 类：退款率高 -> 高退款风险用户
5. 肘部法则
model.inertia_

inertia 越小，说明样本离自己簇中心越近。

但是 K 越大，inertia 一定会下降。

所以不是越小越好，而是看：

从哪个 K 开始下降速度明显变慢。
6. 轮廓系数
score = silhouette_score(self.X_scaled, labels)

轮廓系数越高，通常说明聚类效果越好。

它可以辅助选择 K。

但注意：

轮廓系数不是唯一标准；
最终还要结合业务解释。
7. DBSCAN 聚类
self.dbscan_model = DBSCAN(
    eps=eps,
    min_samples=min_samples
)

DBSCAN 输出的标签中：

-1 表示噪声点。

如果噪声点很多，可能说明：

eps 太小；
min_samples 太大；
数据本身簇结构不适合当前参数。
十六、运行后重点观察什么？
1. 原始数据

重点看这些字段：

annual_spend
order_count
avg_order_amount
discount_rate
return_rate

思考它们分别能表示用户什么行为。

2. 肘部法则曲线

观察 K 从 1 到 10 时 inertia 如何下降。

重点找：

下降速度开始变慢的位置。

因为我们模拟了 4 类用户，所以你可能会看到 K=4 附近比较合理。

3. 轮廓系数曲线

观察哪个 K 的 silhouette_score 较高。

如果 K=4 分数较高，说明分成 4 类可能比较合适。

但如果 K=3 或 K=5 也不错，也要结合业务解释。

4. KMeans 分群画像

输出类似：

kmeans_cluster  annual_spend  order_count  discount_rate  return_rate
0               1200          5            0.12           0.03
1               12000         45           0.08           0.02
2               4500          18           0.35           0.05
3               6500          22           0.18           0.28

你可以解释为：

第 0 类：普通低频用户
第 1 类：高价值高频用户
第 2 类：优惠敏感用户
第 3 类：高退款风险用户

注意：实际 cluster 编号不固定。

可能这次第 0 类是高价值用户，下次第 1 类才是高价值用户。

所以一定要看画像，不要死记编号。

5. DBSCAN 噪声点

观察：

dbscan_cluster = -1

这些是 DBSCAN 认为的噪声点或异常点。

在用户分析中，它们可能代表：

行为特别异常的用户；
极高消费但极少下单用户；
退款率特别高的用户；
和其他人差异很大的用户。
十七、聚类结果如何命名？

聚类算法不会自动给业务名称。

你需要根据每个簇的特征均值命名。

例如：

cluster	特征表现	可命名为
0	消费低、下单少	普通低频用户
1	消费高、下单多、退款低	高价值用户
2	优惠率高、消费中等	优惠敏感用户
3	退款率高、下单中等	高退款风险用户

命名时要遵循：

先看数据画像；
再结合业务理解；
不要只看 cluster 编号。
十八、聚类和用户运营

聚类不是为了分组而分组，而是为了后续策略。

例如：

1. 高价值用户

特征：

消费高；
下单多；
退款低。

运营策略：

会员权益；
专属客服；
新品优先推荐；
高端商品推荐。
2. 普通低频用户

特征：

消费低；
下单少；
活跃度低。

运营策略：

新人券；
低门槛优惠；
基础商品推荐；
提高首次复购。
3. 优惠敏感用户

特征：

优惠率高；
可能更愿意在打折时购买。

运营策略：

优惠券推送；
满减活动；
限时折扣；
促销提醒。
4. 高退款风险用户

特征：

退款率高；
售后风险高。

运营策略：

加强商品说明；
优化售前咨询；
谨慎发放高成本优惠；
分析退款原因。
十九、聚类的常见问题
1. 聚类结果一定对吗？

不一定。

聚类没有标准答案，所以结果需要结合：

业务解释；
可视化；
轮廓系数；
稳定性；
后续策略效果。
2. KMeans 的 K 越大越好吗？

不是。

K 越大，分得越细，inertia 会下降。

但 K 太大可能导致：

每组人数太少；
业务解释困难；
运营策略无法落地。
3. 聚类需要标签吗？

不需要。

但如果有真实业务标签，也可以用来辅助验证聚类效果。

例如：

真实 VIP 用户是否集中在某些簇；
高退款用户是否集中在某些簇。
4. 聚类结果能直接用于预测吗？

聚类本身不是预测模型。

但聚类结果可以作为新特征加入监督学习模型。

例如：

先给用户聚类，得到 user_cluster；
再把 user_cluster 作为特征，预测用户是否复购。

这在实际项目中也很常见。

二十、今日练习
练习 1：解释聚类和分类

请用自己的话解释：

聚类和分类有什么区别？

参考方向：

分类有标签，模型学习已有类别；
聚类没有标签，模型根据样本相似性自动分组。
练习 2：修改 KMeans 的 K 值

把代码中的：

n_clusters=4

分别改成：

n_clusters=2
n_clusters=3
n_clusters=5
n_clusters=6

观察：

分群画像是否更清晰？
每组人数是否合理？
业务解释是否容易？
练习 3：删除标准化试试

把：

self.X_scaled = self.scaler.fit_transform(self.X)

改成：

self.X_scaled = self.X.values

观察聚类结果是否变化。

思考：

为什么年消费金额会影响距离计算？
练习 4：调整 DBSCAN 参数

修改：

eps=0.9
min_samples=5

分别尝试：

eps=0.5
eps=1.2
min_samples=3
min_samples=8

观察：

簇数量是否变化？
噪声点数量是否变化？
练习 5：新增一个特征

在模拟数据中新增一个字段：

coupon_use_rate：优惠券使用率

然后加入：

self.feature_columns = [
    "annual_spend",
    "order_count",
    "avg_order_amount",
    "discount_rate",
    "return_rate",
    "coupon_use_rate"
]

观察聚类画像是否更容易解释优惠敏感用户。

二十一、今日学习检查表
检查项	是否掌握
能理解无监督学习	
能区分分类和聚类	
能理解 KMeans 基本思想	
能理解 KMeans 中 K 的含义	
能理解聚类为什么需要标准化	
能理解肘部法则	
能理解轮廓系数	
能理解 DBSCAN 的 eps 和 min_samples	
能根据聚类画像解释用户群体	
能用聚类做用户分层	
二十二、今日总结

第 15 天最重要的是这句话：

聚类是在没有标签的情况下，根据样本之间的相似性自动分组。

今天你应该掌握：

分类有 y，聚类没有 y；
KMeans 需要指定 K；
KMeans 依赖距离，所以通常需要标准化；
肘部法则和轮廓系数可以辅助选择 K；
DBSCAN 按密度聚类，可以发现噪声点；
聚类结果需要结合业务画像解释。

今天还要特别记住：

聚类输出的 0、1、2、3 本身没有业务意义；
真正重要的是分析每个簇的特征画像，然后给它命名。