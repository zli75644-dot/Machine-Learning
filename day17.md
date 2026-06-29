这里我按前面 20 天计划继续生成“第 17 天”的内容。今天会从表格数据切换到文本数据，核心是把中文句子变成机器学习模型能理解的数字特征。

已思考 13s
第 17 天：文本机器学习入门

前 16 天你主要处理的是结构化数据，比如：

面积、楼层、房间数
年龄、收入、学习时长
订单数、消费金额、退款率
细胞半径、纹理、面积

这些数据本身就是数值或表格字段。

第 17 天开始进入 文本机器学习。

文本数据长这样：

这个商品质量很好，下次还会买
物流太慢了，客服态度也不好
这节课讲得很清楚，代码也能跑通
模型训练失败了，报错信息看不懂

机器学习模型不能直接理解中文句子，所以今天的核心问题是：

如何把文本变成数字？
一、今日学习目标

今天需要掌握 8 个内容：

什么是文本机器学习；
文本为什么要向量化；
什么是中文分词；
什么是词袋模型 Bag of Words；
什么是 TF-IDF；
什么是朴素贝叶斯；
如何做中文评论情感分类；
如何用 Pipeline 封装文本分类流程。
二、什么是文本机器学习？

文本机器学习就是让模型从文本中学习规律。

常见任务包括：

评论情感分类：好评 / 差评
垃圾邮件识别：垃圾邮件 / 正常邮件
新闻分类：体育 / 财经 / 科技 / 娱乐
问题分类：售后问题 / 物流问题 / 支付问题
舆情分析：正面 / 负面 / 中性
文本聚类：把相似文章分到一起
关键词提取：找出文章核心词

例如：

文本	标签
商品质量很好，物流也快	好评
太差了，包装破损，客服也不回复	差评
课程讲得很清楚，适合初学者	好评
代码跑不通，讲解也很乱	差评

这就是一个监督学习任务。

其中：

X = 文本内容
y = 文本标签
三、文本为什么不能直接进模型？

模型能处理的是数字矩阵。

例如表格数据：

学习时长	出勤率	是否挂科
5.0	0.95	0
1.0	0.50	1

这种可以直接变成：

X = [[5.0, 0.95],
     [1.0, 0.50]]

但是文本是：

这个商品质量很好

模型不能直接计算这句话。

所以需要把文本转换成数字向量。

这个过程叫：

文本向量化 Text Vectorization
四、中文分词

英文文本天然有空格：

this product is very good

可以直接按空格拆成：

this / product / is / very / good

但中文没有天然空格：

这个商品质量很好

如果按字切：

这 / 个 / 商 / 品 / 质 / 量 / 很 / 好

信息太碎。

更合理的是分词：

这个 / 商品 / 质量 / 很好

所以中文文本处理通常需要中文分词。

常用工具：

jieba

安装：

pip install jieba

示例：

import jieba

text = "这个商品质量很好，下次还会买"
words = jieba.lcut(text)

print(words)

可能输出：

['这个', '商品', '质量', '很', '好', '，', '下次', '还会', '买']
五、停用词 Stop Words

停用词是一些很常见但信息量不大的词。

例如：

的
了
是
我
你
他
这个
那个
一个

在很多文本分类任务中，这些词对判断类别帮助不大。

例如：

这个商品质量很好
这个商品质量很差

真正影响情感的是：

很好
很差

不是：

这个
商品

所以有时候会过滤停用词。

不过入门阶段你先掌握：

分词 → 向量化 → 训练模型

停用词后面可以慢慢优化。

六、词袋模型 Bag of Words

词袋模型是最基础的文本向量化方法。

它的思想是：

不考虑词语顺序，只统计每个词出现了多少次。

例如有两句话：

句子1：商品 质量 很好
句子2：商品 质量 很差

词表是：

商品，质量，很好，很差

转换成数字：

句子	商品	质量	很好	很差
句子1	1	1	1	0
句子2	1	1	0	1

这就是词袋模型。

在 sklearn 中可以用：

CountVectorizer
七、TF-IDF

词袋模型只统计词频，但它有一个问题：

常见词出现很多次，不一定重要。

例如：

商品
这个
感觉
真的

这些词可能很多评论里都有。

而下面这些词更能区分类别：

好用
满意
推荐
垃圾
失望
破损

TF-IDF 的思想是：

一个词在当前文本中出现越多，越重要；
但如果它在所有文本中都很常见，就降低它的重要性。

你可以这样理解：

TF：这个词在当前文本中出现得多不多；
IDF：这个词在所有文本中是不是比较稀有；
TF-IDF：既看当前文本中的频率，也看全局区分度。

在 sklearn 中可以用：

TfidfVectorizer
八、朴素贝叶斯 Naive Bayes

朴素贝叶斯是文本分类中非常经典的模型。

它特别适合：

垃圾邮件分类
评论情感分类
新闻分类
短文本分类

它的核心思想是：

根据词语出现情况，计算文本属于某个类别的概率。

例如：

“好用”“满意”“推荐” 经常出现在好评中；
“差劲”“失望”“破损” 经常出现在差评中。

那么新评论：

商品很好用，非常满意

模型就更可能判断为：

好评

在 sklearn 中常用：

MultinomialNB
九、今日案例：中文评论情感分类

今天做一个简单的中文评论情感分类器。

任务是：

输入一条中文评论；
判断它是好评还是差评。

标签设计：

0：差评
1：好评

流程是：

构造评论数据
    ↓
中文分词
    ↓
TF-IDF 向量化
    ↓
训练朴素贝叶斯模型
    ↓
评估模型
    ↓
预测新评论
十、今日完整代码

新建文件：

day17_text_machine_learning.py

代码如下：

"""
第 17 天：文本机器学习入门。

本案例使用模拟中文评论数据，完成一个简单的中文情感分类任务。

核心目标：
1. 理解文本不能直接进入模型
2. 理解中文分词
3. 理解 CountVectorizer 词袋模型
4. 理解 TfidfVectorizer
5. 理解朴素贝叶斯文本分类
6. 使用 Pipeline 封装文本分类流程
"""

import re
import jieba
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ChineseReviewDataBuilder:
    """
    中文评论数据构造类。

    :return: 无返回值
    """

    def __init__(self):
        """
        初始化中文评论数据构造类。

        :return: 无返回值
        """
        self.df = None

    def build_data(self) -> pd.DataFrame:
        """
        构造模拟中文评论数据。

        label:
        0 表示差评
        1 表示好评

        :return: 评论数据 DataFrame
        """
        positive_reviews = [
            "这个商品质量很好，下次还会继续购买",
            "物流速度很快，包装也很完整",
            "客服态度很好，问题解决得很及时",
            "课程讲得很清楚，适合初学者学习",
            "这个模型效果不错，准确率比之前高",
            "代码结构清晰，注释也比较完整",
            "体验很好，整体非常满意",
            "商品做工精细，性价比很高",
            "发货很快，收到后没有任何问题",
            "老师讲解很耐心，知识点很容易理解",
            "界面设计简洁，使用起来很方便",
            "功能很实用，运行也比较稳定",
            "服务态度不错，回复速度很快",
            "这次购物体验很好，值得推荐",
            "内容非常详细，对学习很有帮助",
            "程序运行成功，结果也符合预期",
            "文档写得清楚，安装过程很顺利",
            "模型训练速度快，预测效果也不错",
            "商品颜色好看，质量也让人满意",
            "整体效果很好，没有明显缺点",
            "讲得通俗易懂，例子也很贴近实际",
            "系统响应速度快，操作体验很好",
            "这家店服务很好，售后也很负责",
            "包装严实，没有破损，质量可靠",
            "价格合理，质量也不错，推荐购买",
            "学习资料很系统，适合复习使用",
            "实验步骤清楚，跟着做可以跑通",
            "项目结构合理，代码维护起来方便",
            "运行稳定，没有出现明显报错",
            "这次更新之后体验提升明显"
        ]

        negative_reviews = [
            "这个商品质量很差，用了两天就坏了",
            "物流太慢了，等了很久才收到",
            "客服态度不好，问题一直没有解决",
            "课程讲得很乱，很多地方听不懂",
            "这个模型效果很差，准确率非常低",
            "代码结构混乱，注释也不清楚",
            "体验很差，整体非常失望",
            "商品做工粗糙，性价比很低",
            "发货太慢，包装还有破损",
            "老师讲解不清楚，知识点很难理解",
            "界面设计复杂，使用起来很麻烦",
            "功能不好用，运行也不稳定",
            "服务态度很差，回复速度太慢",
            "这次购物体验很糟糕，不推荐",
            "内容太简单，对学习帮助不大",
            "程序一直报错，结果也不符合预期",
            "文档写得很乱，安装过程不顺利",
            "模型训练很慢，预测效果也不好",
            "商品颜色难看，质量也让人失望",
            "整体效果很差，问题比较多",
            "讲得太抽象，例子也不贴近实际",
            "系统响应速度慢，操作体验很差",
            "这家店服务很差，售后也不负责",
            "包装很差，收到时已经破损",
            "价格偏高，质量也不好，不建议购买",
            "学习资料不系统，复习起来很困难",
            "实验步骤不清楚，跟着做也跑不通",
            "项目结构混乱，代码维护起来很麻烦",
            "运行不稳定，经常出现明显报错",
            "这次更新之后体验反而变差了"
        ]

        rows = []

        for text in positive_reviews:
            rows.append({
                "text": text,
                "label": 1,
                "label_name": "好评"
            })

        for text in negative_reviews:
            rows.append({
                "text": text,
                "label": 0,
                "label_name": "差评"
            })

        self.df = pd.DataFrame(rows)
        return self.df


class ChineseTextPreprocessor:
    """
    中文文本预处理类。

    :return: 无返回值
    """

    def __init__(self):
        """
        初始化中文文本预处理类。

        :return: 无返回值
        """
        self.stop_words = {
            "的", "了", "是", "我", "你", "他", "她", "它",
            "这个", "那个", "一个", "很", "也", "都", "就",
            "在", "和", "对", "但", "还", "比较", "非常"
        }

    def clean_text(self, text: str) -> str:
        """
        清洗文本。

        当前主要去除多余空白字符。
        也可以根据需要去除特殊符号、数字、英文等。

        :param text: 原始文本
        :return: 清洗后的文本
        """
        text = str(text)
        text = re.sub(r"\s+", "", text)
        return text

    def jieba_tokenizer(self, text: str) -> list:
        """
        使用 jieba 对中文文本分词。

        :param text: 原始文本
        :return: 分词后的词列表
        """
        text = self.clean_text(text)

        words = jieba.lcut(text)

        filtered_words = []

        for word in words:
            word = word.strip()

            if not word:
                continue

            if word in self.stop_words:
                continue

            if re.fullmatch(r"[，。！？、,.!?；;：:（）()]", word):
                continue

            filtered_words.append(word)

        return filtered_words


class TextClassificationDemo:
    """
    中文文本分类演示类。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, test_size: float = 0.25, random_state: int = 42):
        """
        初始化中文文本分类演示类。

        :param test_size: 测试集比例
        :param random_state: 随机种子
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state

        self.df = None

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.text_preprocessor = ChineseTextPreprocessor()

        self.model_pipeline = None
        self.y_pred = None

    def load_data(self):
        """
        加载中文评论数据。

        :return: 无返回值
        """
        builder = ChineseReviewDataBuilder()
        self.df = builder.build_data()

    def show_data_info(self):
        """
        打印数据基本信息。

        :return: 无返回值
        """
        print("========== 中文评论数据 ==========")
        print(self.df.head(10))

        print("\n========== 数据形状 ==========")
        print(self.df.shape)

        print("\n========== 标签分布 ==========")
        print(self.df["label_name"].value_counts())

    def show_tokenization_examples(self):
        """
        展示中文分词效果。

        :return: 无返回值
        """
        print("\n========== 中文分词示例 ==========")

        examples = [
            "这个商品质量很好，下次还会继续购买",
            "物流太慢了，客服态度也不好",
            "课程讲得很清楚，适合初学者学习"
        ]

        for text in examples:
            words = self.text_preprocessor.jieba_tokenizer(text)
            print("原文：", text)
            print("分词：", words)
            print("-" * 50)

    def prepare_features_and_label(self):
        """
        准备文本特征 X 和标签 y。

        :return: 无返回值
        """
        self.X = self.df["text"]
        self.y = self.df["label"]

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
        print("训练集数量：", len(self.X_train))
        print("测试集数量：", len(self.X_test))

    def build_tfidf_nb_pipeline(self):
        """
        构建 TF-IDF + 朴素贝叶斯分类 Pipeline。

        TfidfVectorizer:
            把中文文本转换成 TF-IDF 数字特征。

        MultinomialNB:
            适合文本分类的朴素贝叶斯模型。

        :return: 无返回值
        """
        vectorizer = TfidfVectorizer(
            tokenizer=self.text_preprocessor.jieba_tokenizer,
            token_pattern=None,
            max_features=1000
        )

        classifier = MultinomialNB()

        self.model_pipeline = Pipeline(
            steps=[
                ("vectorizer", vectorizer),
                ("classifier", classifier)
            ]
        )

    def train_model(self):
        """
        训练文本分类模型。

        :return: 无返回值
        """
        self.model_pipeline.fit(self.X_train, self.y_train)
        print("\n文本分类模型训练完成。")

    def predict_test_data(self):
        """
        使用测试集进行预测。

        :return: 无返回值
        """
        self.y_pred = self.model_pipeline.predict(self.X_test)

    def evaluate_model(self):
        """
        评估文本分类模型。

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
            target_names=["差评", "好评"]
        ))

    def show_prediction_detail(self):
        """
        展示测试集部分预测结果。

        :return: 无返回值
        """
        result_df = pd.DataFrame({
            "text": self.X_test.values,
            "真实标签": self.y_test.values,
            "预测标签": self.y_pred
        })

        result_df["真实情感"] = result_df["真实标签"].map({
            0: "差评",
            1: "好评"
        })

        result_df["预测情感"] = result_df["预测标签"].map({
            0: "差评",
            1: "好评"
        })

        result_df["是否预测正确"] = result_df["真实标签"] == result_df["预测标签"]

        print("\n========== 测试集预测详情 ==========")
        print(result_df)

    def show_top_tfidf_words(self, top_n: int = 20):
        """
        展示 TF-IDF 词表中的部分词语。

        :param top_n: 展示前多少个词
        :return: 无返回值
        """
        vectorizer = self.model_pipeline.named_steps["vectorizer"]
        feature_names = vectorizer.get_feature_names_out()

        print(f"\n========== TF-IDF 词表前 {top_n} 个词 ==========")
        print(feature_names[:top_n])

        print("\n词表总数量：", len(feature_names))

    def predict_new_reviews(self):
        """
        预测新的中文评论。

        :return: 无返回值
        """
        new_reviews = [
            "这个商品很好用，质量不错，还会再买",
            "太失望了，物流慢，包装还破了",
            "老师讲得很清楚，代码也能运行成功",
            "系统经常报错，使用体验很差",
            "价格合理，功能实用，整体满意",
            "文档太乱了，安装过程一直失败"
        ]

        pred_labels = self.model_pipeline.predict(new_reviews)
        pred_proba = self.model_pipeline.predict_proba(new_reviews)

        result_df = pd.DataFrame({
            "评论": new_reviews,
            "预测标签": pred_labels,
            "预测情感": ["好评" if label == 1 else "差评" for label in pred_labels],
            "差评概率": pred_proba[:, 0],
            "好评概率": pred_proba[:, 1]
        })

        print("\n========== 新评论预测结果 ==========")
        print(result_df)

    def run(self):
        """
        运行完整中文文本分类流程。

        :return: 无返回值
        """
        print("========== 第 17 天：文本机器学习入门 ==========")

        self.load_data()
        self.show_data_info()
        self.show_tokenization_examples()

        self.prepare_features_and_label()
        self.split_data()

        self.build_tfidf_nb_pipeline()
        self.train_model()
        self.predict_test_data()

        self.evaluate_model()
        self.show_prediction_detail()
        self.show_top_tfidf_words(top_n=30)
        self.predict_new_reviews()


class TextVectorizationComparison:
    """
    文本向量化方法对比类。

    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, random_state: int = 42):
        """
        初始化文本向量化方法对比类。

        :param random_state: 随机种子
        :return: 无返回值
        """
        self.random_state = random_state
        self.text_preprocessor = ChineseTextPreprocessor()

    def build_models(self) -> dict:
        """
        构建不同文本向量化和分类器组合。

        :return: 模型字典
        """
        models = {
            "CountVectorizer + NaiveBayes": Pipeline(
                steps=[
                    (
                        "vectorizer",
                        CountVectorizer(
                            tokenizer=self.text_preprocessor.jieba_tokenizer,
                            token_pattern=None,
                            max_features=1000
                        )
                    ),
                    ("classifier", MultinomialNB())
                ]
            ),
            "TfidfVectorizer + NaiveBayes": Pipeline(
                steps=[
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            tokenizer=self.text_preprocessor.jieba_tokenizer,
                            token_pattern=None,
                            max_features=1000
                        )
                    ),
                    ("classifier", MultinomialNB())
                ]
            ),
            "TfidfVectorizer + LogisticRegression": Pipeline(
                steps=[
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            tokenizer=self.text_preprocessor.jieba_tokenizer,
                            token_pattern=None,
                            max_features=1000
                        )
                    ),
                    ("classifier", LogisticRegression(max_iter=1000))
                ]
            )
        }

        return models

    def compare_models(
        self,
        X_train: pd.Series,
        X_test: pd.Series,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        对比不同文本向量化和模型组合效果。

        :param X_train: 训练集文本
        :param X_test: 测试集文本
        :param y_train: 训练集标签
        :param y_test: 测试集标签
        :return: 对比结果 DataFrame
        """
        models = self.build_models()

        results = []

        print("\n========== 文本向量化方法对比 ==========")

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            results.append({
                "model": model_name,
                "accuracy": accuracy
            })

            print(f"{model_name} Accuracy：{accuracy:.4f}")

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(by="accuracy", ascending=False)

        print("\n========== 对比结果 ==========")
        print(result_df)

        return result_df


def main():
    """
    主函数。

    :return: 无返回值
    """
    demo = TextClassificationDemo(
        test_size=0.25,
        random_state=42
    )

    demo.run()

    comparison = TextVectorizationComparison(
        random_state=42
    )

    comparison.compare_models(
        X_train=demo.X_train,
        X_test=demo.X_test,
        y_train=demo.y_train,
        y_test=demo.y_test
    )


if __name__ == "__main__":
    main()
十一、代码重点解释
1. 数据格式

数据中最重要的两列是：

"text"
"label"

其中：

text：中文评论
label：情感标签，0 是差评，1 是好评

这和前面的监督学习一样：

X = df["text"]
y = df["label"]

不同的是，X 现在不是数值，而是文本。

2. 中文分词函数
def jieba_tokenizer(self, text: str) -> list:

这个函数的作用是：

输入一句中文；
清洗文本；
使用 jieba 分词；
去掉停用词和标点；
输出词语列表。

例如：

这个商品质量很好，下次还会继续购买

可能变成：

商品 / 质量 / 好 / 下次 / 继续 / 购买
3. TF-IDF 向量化
TfidfVectorizer(
    tokenizer=self.text_preprocessor.jieba_tokenizer,
    token_pattern=None,
    max_features=1000
)

解释：

tokenizer：指定中文分词函数；
token_pattern=None：告诉 sklearn 不使用默认英文分词规则；
max_features=1000：最多保留 1000 个重要词。

这一步会把中文文本变成数字矩阵。

4. 朴素贝叶斯分类器
classifier = MultinomialNB()

它常用于文本分类。

原因是文本向量经常是：

词频
TF-IDF
稀疏矩阵

朴素贝叶斯在这类任务中简单、快速、效果通常不错。

5. Pipeline 封装
self.model_pipeline = Pipeline(
    steps=[
        ("vectorizer", vectorizer),
        ("classifier", classifier)
    ]
)

这表示：

先进行 TF-IDF 向量化；
再训练朴素贝叶斯模型。

以后预测新文本时：

self.model_pipeline.predict(new_reviews)

会自动完成：

中文分词
TF-IDF 转换
模型预测
十二、运行后重点观察什么？
1. 分词结果

你要看分词是否合理。

例如：

课程讲得很清楚，适合初学者学习

理想分词可能是：

课程 / 讲得 / 清楚 / 适合 / 初学者 / 学习

如果分词效果很差，后面的文本分类效果也会受影响。

2. TF-IDF 词表数量

代码会输出：

词表总数量

例如：

词表总数量：160

意思是模型最终把文本转换成了 160 个词特征。

3. 模型评估结果

重点看：

Accuracy
混淆矩阵
classification_report

由于当前数据集是模拟小数据，分数可能比较高，但这不代表真实项目一定这么好。

真实项目中数据会更复杂：

表达更丰富；
噪声更多；
标签可能不准确；
好评差评可能不均衡。
4. 新评论预测

例如：

这个商品很好用，质量不错，还会再买

模型应该更倾向预测为：

好评

例如：

太失望了，物流慢，包装还破了

模型应该更倾向预测为：

差评

你要观察模型预测是否符合直觉。

十三、词袋模型和 TF-IDF 的区别
对比项	CountVectorizer	TfidfVectorizer
中文名	词袋模型 / 词频模型	TF-IDF 模型
关注点	词出现了多少次	词是否有区分度
常见输出	词频	权重
是否降低常见词影响	否	是
适合场景	简单文本分类	更常用的文本分类基线

简单理解：

CountVectorizer：这个词出现了几次；
TfidfVectorizer：这个词对区分这篇文本有多重要。
十四、为什么文本特征通常是稀疏矩阵？

假设词表有 5000 个词。

一条评论可能只有 20 个词。

那么这一条评论对应的向量长度是 5000，但大部分位置都是 0。

例如：

[0, 0, 1, 0, 0, 0, 2, 0, ..., 0]

这种大部分值为 0 的矩阵叫：

稀疏矩阵 Sparse Matrix

文本机器学习中很常见。

十五、文本分类的一般流程

真实项目中，文本分类一般流程是：

收集文本数据
    ↓
人工标注标签
    ↓
清洗文本
    ↓
中文分词
    ↓
去停用词
    ↓
文本向量化
    ↓
训练分类模型
    ↓
评估模型
    ↓
预测新文本

对应到 sklearn：

jieba 分词
    ↓
TfidfVectorizer
    ↓
MultinomialNB / LogisticRegression
    ↓
classification_report
十六、文本机器学习常见模型
1. 朴素贝叶斯

适合：

短文本分类；
垃圾邮件分类；
情感分类入门；
小数据基线模型。

优点：

速度快；
实现简单；
对文本任务效果不错。
2. 逻辑回归

适合：

文本二分类；
情感分类；
舆情分类；
垃圾邮件分类。

优点：

稳定；
可解释性较好；
经常作为强基线模型。
3. SVM

适合：

高维稀疏文本特征；
中小规模文本分类。

优点：

在传统文本分类中效果很好。
4. 深度学习模型

例如：

TextCNN
RNN
LSTM
BERT
RoBERTa
ERNIE

这些适合更复杂的 NLP 任务，但入门阶段先掌握：

分词 + TF-IDF + 传统分类器

非常重要。

十七、中文文本处理常见问题
1. 分词不准

例如：

机器学习很好玩

可能分成：

机器 / 学习 / 很 / 好玩

也可能分成：

机器学习 / 很 / 好玩

分词结果会影响特征。

2. 同义词问题

例如：

很好
不错
满意
可以
推荐

这些都可能表达正面情感。

但传统 TF-IDF 不知道它们语义相近。

3. 否定词问题

例如：

好
不好
不是很好
没有想象中好

传统词袋模型可能只看到“好”，但不一定理解否定语义。

4. 讽刺和反话

例如：

真是太“好”了，用一天就坏

传统模型很难理解这种语境。

5. 领域词问题

不同场景有不同关键词。

例如电商：

物流
客服
包装
退货
质量

教学评价：

讲解
代码
实验
知识点
作业

所以文本分类要结合具体领域构建数据。

十八、真实项目中如何提升文本分类效果？

可以从这几个方向优化：

增加训练数据；
提高标签质量；
优化中文分词词典；
加入领域停用词；
使用 n-gram 特征；
调整 max_features；
尝试 LogisticRegression、SVM；
处理类别不平衡；
使用更强的预训练模型。

例如加入 n-gram：

TfidfVectorizer(
    tokenizer=self.text_preprocessor.jieba_tokenizer,
    token_pattern=None,
    ngram_range=(1, 2),
    max_features=3000
)

含义是：

不仅看单个词；
也看连续两个词组成的短语。

比如：

质量 好
物流 慢
客服 差
值得 推荐

这些短语可能比单个词更有区分度。

十九、今日练习
练习 1：解释文本向量化

请用自己的话解释：

为什么文本要先向量化，才能进入机器学习模型？

参考方向：

模型只能处理数字，不能直接理解中文句子，所以需要通过 CountVectorizer 或 TF-IDF 把文本转换成数字特征。
练习 2：观察中文分词

自己写 5 条评论，例如：

这个课程讲得很清楚
代码跑不通，感觉很崩溃
老师讲解很耐心
物流太慢了
商品质量不错

调用：

demo.text_preprocessor.jieba_tokenizer(text)

观察分词结果是否合理。

练习 3：比较 CountVectorizer 和 TfidfVectorizer

运行代码中的：

TextVectorizationComparison

观察：

CountVectorizer + NaiveBayes
TfidfVectorizer + NaiveBayes
TfidfVectorizer + LogisticRegression

哪个效果更好。

练习 4：加入 n-gram

把：

TfidfVectorizer(
    tokenizer=self.text_preprocessor.jieba_tokenizer,
    token_pattern=None,
    max_features=1000
)

改成：

TfidfVectorizer(
    tokenizer=self.text_preprocessor.jieba_tokenizer,
    token_pattern=None,
    ngram_range=(1, 2),
    max_features=2000
)

观察模型效果是否变化。

练习 5：增加自己的数据

在 positive_reviews 和 negative_reviews 中各增加 10 条你自己的评论。

例如好评：

老师讲得很细，适合零基础学习
实验过程很完整，代码能直接运行

差评：

讲解太跳跃，很多地方跟不上
代码报错太多，文档也不清楚

观察模型预测效果是否更稳定。

二十、今日学习检查表
检查项	是否掌握
能理解文本机器学习的任务	
能理解文本为什么要向量化	
能使用 jieba 做中文分词	
能理解停用词	
能理解 CountVectorizer	
能理解 TF-IDF	
能理解朴素贝叶斯用于文本分类	
能使用 Pipeline 封装文本分类流程	
能训练中文评论情感分类模型	
能预测新的中文评论情感	
二十一、今日总结

第 17 天最重要的是这句话：

文本机器学习的核心是：先把文本变成数字向量，再使用机器学习模型分类。

今天你应该掌握这条流程：

中文评论
    ↓
文本清洗
    ↓
jieba 分词
    ↓
TF-IDF 向量化
    ↓
朴素贝叶斯 / 逻辑回归
    ↓
情感分类结果

你现在应该能理解下面这段代码：

model = Pipeline(
    steps=[
        ("vectorizer", TfidfVectorizer(
            tokenizer=jieba_tokenizer,
            token_pattern=None,
            max_features=1000
        )),
        ("classifier", MultinomialNB())
    ]
)

它背后的含义是：

先把中文句子分词；
再把词转换成 TF-IDF 数字特征；
最后用朴素贝叶斯判断好评或差评。