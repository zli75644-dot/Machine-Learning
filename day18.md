第 18 天会把前面训练好的模型“落地使用”：不只是训练和评估，还要保存模型、重新加载模型，并做成可以被程序调用的预测接口。

已思考 12s
第 18 天：模型保存与部署

前 17 天你已经会训练很多模型了：

逻辑回归
KNN
决策树
随机森林
GBDT
KMeans
PCA
文本分类模型

但是只会训练模型还不够。真实项目中，模型训练完之后还要解决一个问题：

怎么把训练好的模型保存起来，并在以后直接使用？

如果每次预测都重新训练模型，会非常低效。

所以第 18 天重点学习：

模型保存
模型加载
Pipeline 保存
单条数据预测
Flask 接口部署
FastAPI 接口部署

scikit-learn 官方文档也专门把“模型持久化”作为一个独立主题：训练好模型后，通常需要把模型保存起来，以便以后不用重新训练就能再次使用。

一、今日学习目标

今天需要掌握 8 个内容：

为什么要保存模型；
使用 joblib.dump() 保存模型；
使用 joblib.load() 加载模型；
为什么推荐保存整个 Pipeline；
如何预测单条新数据；
如何用 Flask 写一个预测接口；
如何用 FastAPI 写一个预测接口；
理解模型部署的基本流程。
二、为什么要保存模型？

前面训练模型时，流程一般是：

加载数据
    ↓
数据预处理
    ↓
训练模型
    ↓
评估模型
    ↓
预测新数据

但真实项目中，训练模型可能需要很长时间。

例如：

几万条数据
几十万条数据
几百万条数据
大量特征工程
复杂模型调参

如果每次预测都重新训练模型，就很浪费。

正确做法是：

训练一次模型
    ↓
保存模型文件
    ↓
以后直接加载模型
    ↓
输入新数据
    ↓
输出预测结果
三、什么是模型持久化？

模型持久化就是：

把训练好的模型保存到硬盘文件中。

例如保存成：

student_model.joblib
text_sentiment_model.joblib
breast_cancer_pipeline.joblib

以后需要预测时：

直接加载这个文件；
不用重新训练。
四、为什么推荐保存 Pipeline？

这是今天最重要的意识。

如果你只保存模型：

model = LogisticRegression()

但是没有保存：

StandardScaler
OneHotEncoder
TfidfVectorizer
PCA
缺失值填充器

以后预测新数据时就会出问题。

因为模型训练时用的是处理后的数据。

例如：

原始文本
    ↓
jieba 分词
    ↓
TF-IDF
    ↓
朴素贝叶斯模型

如果只保存朴素贝叶斯模型，不保存 TF-IDF 向量化器，那新文本就无法转换成模型需要的数字特征。

所以真实项目中更推荐保存：

完整 Pipeline

也就是：

预处理步骤 + 模型

一起保存。

五、joblib 是什么？

joblib 是 Python 中常用的模型保存工具。

在 scikit-learn 项目中，经常用它保存模型或 Pipeline。

常见用法：

import joblib

joblib.dump(model, "model.joblib")
loaded_model = joblib.load("model.joblib")

入门阶段你先记住：

dump 是保存；
load 是加载。
六、今日案例：保存一个中文情感分类模型

今天延续第 17 天的中文评论情感分类。

流程是：

构造中文评论数据
    ↓
jieba 中文分词
    ↓
TF-IDF 向量化
    ↓
逻辑回归分类
    ↓
保存整个 Pipeline
    ↓
重新加载 Pipeline
    ↓
预测新评论
    ↓
提供 Web API 接口
七、今日项目文件结构

建议新建一个文件夹：

day18_model_deployment/

里面放 4 个文件：

day18_train_and_save_model.py
day18_load_and_predict.py
day18_flask_api.py
day18_fastapi_api.py

分别对应：

训练并保存模型；
加载模型并预测；
Flask 接口部署；
FastAPI 接口部署。
八、安装依赖
pip install jieba joblib scikit-learn pandas flask fastapi uvicorn

说明：

jieba：中文分词
joblib：保存和加载模型
scikit-learn：训练机器学习模型
pandas：处理数据
flask：构建简单 Web API
fastapi：构建现代 API 服务
uvicorn：运行 FastAPI 服务

Flask 官方文档将 Flask 定位为轻量级 WSGI Web 应用框架，适合快速开始并能扩展到更复杂应用；FastAPI 官方文档将其定位为基于 Python 类型提示构建 API 的现代 Web 框架，并支持自动生成 OpenAPI 相关接口文档。

九、文件 1：训练并保存模型

文件名：

day18_train_and_save_model.py

代码如下：

"""
第 18 天：训练并保存中文情感分类模型。

本文件完成：
1. 构造中文评论数据
2. 中文文本清洗和分词
3. 构建 TF-IDF + 逻辑回归 Pipeline
4. 训练模型
5. 评估模型
6. 使用 joblib 保存完整 Pipeline

重点：
保存模型时，建议保存完整 Pipeline，而不是只保存分类器。
"""

import os
import re
import jieba
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ChineseTextPreprocessor:
    """
    中文文本预处理类。

    主要负责：
    1. 清洗文本
    2. jieba 中文分词
    3. 去除停用词和标点符号

    :return: 无返回值
    """

    def __init__(self):
        """
        初始化中文文本预处理器。

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

        :param text: 原始文本
        :return: 清洗后的文本
        """
        text = str(text)
        text = re.sub(r"\s+", "", text)
        return text

    def jieba_tokenizer(self, text: str) -> list:
        """
        使用 jieba 进行中文分词。

        :param text: 原始中文文本
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

            if re.fullmatch(r"[，。！？、,.!?；;：:（）()【】\[\]{}]", word):
                continue

            filtered_words.append(word)

        return filtered_words


class ReviewDataBuilder:
    """
    中文评论数据构造类。

    :return: 无返回值
    """

    def __init__(self):
        """
        初始化评论数据构造器。

        :return: 无返回值
        """
        self.df = None

    def build_data(self) -> pd.DataFrame:
        """
        构造模拟中文评论情感分类数据。

        label:
        0 表示差评
        1 表示好评

        :return: 评论 DataFrame
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

        for review in positive_reviews:
            rows.append({
                "text": review,
                "label": 1,
                "label_name": "好评"
            })

        for review in negative_reviews:
            rows.append({
                "text": review,
                "label": 0,
                "label_name": "差评"
            })

        self.df = pd.DataFrame(rows)
        return self.df


class SentimentModelTrainer:
    """
    中文情感分类模型训练类。

    :param model_path: 模型保存路径
    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(
        self,
        model_path: str = "sentiment_pipeline.joblib",
        test_size: float = 0.25,
        random_state: int = 42
    ):
        """
        初始化模型训练器。

        :param model_path: 模型保存路径
        :param test_size: 测试集比例
        :param random_state: 随机种子
        :return: 无返回值
        """
        self.model_path = model_path
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
        self.pipeline = None
        self.y_pred = None

    def load_data(self):
        """
        加载训练数据。

        :return: 无返回值
        """
        builder = ReviewDataBuilder()
        self.df = builder.build_data()

        print("========== 数据集信息 ==========")
        print(self.df.head())
        print("\n数据形状：", self.df.shape)
        print("\n标签分布：")
        print(self.df["label_name"].value_counts())

    def prepare_features_and_label(self):
        """
        准备特征 X 和标签 y。

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

        print("\n========== 数据集划分 ==========")
        print("训练集数量：", len(self.X_train))
        print("测试集数量：", len(self.X_test))

    def build_pipeline(self):
        """
        构建完整文本分类 Pipeline。

        Pipeline 包含：
        1. TfidfVectorizer：文本转数字特征
        2. LogisticRegression：分类模型

        :return: 无返回值
        """
        vectorizer = TfidfVectorizer(
            tokenizer=self.text_preprocessor.jieba_tokenizer,
            token_pattern=None,
            ngram_range=(1, 2),
            max_features=3000
        )

        classifier = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state
        )

        self.pipeline = Pipeline(
            steps=[
                ("vectorizer", vectorizer),
                ("classifier", classifier)
            ]
        )

    def train_model(self):
        """
        训练模型。

        :return: 无返回值
        """
        self.pipeline.fit(self.X_train, self.y_train)
        print("\n模型训练完成。")

    def evaluate_model(self):
        """
        评估模型效果。

        :return: 无返回值
        """
        self.y_pred = self.pipeline.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, self.y_pred)

        print("\n========== 模型评估 ==========")
        print(f"Accuracy：{accuracy:.4f}")

        print("\n混淆矩阵：")
        print(confusion_matrix(self.y_test, self.y_pred))

        print("\n分类报告：")
        print(classification_report(
            self.y_test,
            self.y_pred,
            target_names=["差评", "好评"]
        ))

    def save_model(self):
        """
        使用 joblib 保存完整 Pipeline。

        :return: 无返回值
        """
        joblib.dump(self.pipeline, self.model_path)

        print("\n========== 模型保存完成 ==========")
        print("模型文件路径：", os.path.abspath(self.model_path))

    def run(self):
        """
        运行完整训练和保存流程。

        :return: 无返回值
        """
        print("========== 第 18 天：训练并保存模型 ==========")

        self.load_data()
        self.prepare_features_and_label()
        self.split_data()

        self.build_pipeline()
        self.train_model()
        self.evaluate_model()
        self.save_model()


def main():
    """
    主函数。

    :return: 无返回值
    """
    trainer = SentimentModelTrainer(
        model_path="sentiment_pipeline.joblib",
        test_size=0.25,
        random_state=42
    )

    trainer.run()


if __name__ == "__main__":
    main()

运行：

python day18_train_and_save_model.py

运行后会生成：

sentiment_pipeline.joblib
十、文件 2：加载模型并预测

文件名：

day18_load_and_predict.py

代码如下：

"""
第 18 天：加载模型并预测新文本。

本文件完成：
1. 使用 joblib 加载已保存的 Pipeline
2. 输入新的中文评论
3. 输出预测标签和预测概率

注意：
加载模型时，需要保证训练时用到的自定义分词类和函数在当前环境中可用。
"""

import re
import jieba
import joblib
import pandas as pd


class ChineseTextPreprocessor:
    """
    中文文本预处理类。

    注意：
    这个类需要和训练阶段保持一致。
    因为保存的 Pipeline 中引用了这个 tokenizer。
    """

    def __init__(self):
        """
        初始化中文文本预处理器。

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

        :param text: 原始文本
        :return: 清洗后的文本
        """
        text = str(text)
        text = re.sub(r"\s+", "", text)
        return text

    def jieba_tokenizer(self, text: str) -> list:
        """
        jieba 中文分词。

        :param text: 原始文本
        :return: 分词结果
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

            if re.fullmatch(r"[，。！？、,.!?；;：:（）()【】\[\]{}]", word):
                continue

            filtered_words.append(word)

        return filtered_words


class SentimentPredictor:
    """
    情感分类预测器。

    :param model_path: 模型文件路径
    :return: 无返回值
    """

    def __init__(self, model_path: str = "sentiment_pipeline.joblib"):
        """
        初始化预测器。

        :param model_path: 模型文件路径
        :return: 无返回值
        """
        self.model_path = model_path
        self.pipeline = None

    def load_model(self):
        """
        加载已保存的模型 Pipeline。

        :return: 无返回值
        """
        self.pipeline = joblib.load(self.model_path)
        print("模型加载完成：", self.model_path)

    def predict(self, texts: list) -> pd.DataFrame:
        """
        批量预测中文评论情感。

        :param texts: 中文评论列表
        :return: 预测结果 DataFrame
        """
        if self.pipeline is None:
            self.load_model()

        pred_labels = self.pipeline.predict(texts)
        pred_proba = self.pipeline.predict_proba(texts)

        result_df = pd.DataFrame({
            "text": texts,
            "pred_label": pred_labels,
            "pred_name": ["好评" if label == 1 else "差评" for label in pred_labels],
            "negative_probability": pred_proba[:, 0],
            "positive_probability": pred_proba[:, 1]
        })

        return result_df


def main():
    """
    主函数。

    :return: 无返回值
    """
    predictor = SentimentPredictor(
        model_path="sentiment_pipeline.joblib"
    )

    new_texts = [
        "这个商品质量很好，物流也很快",
        "太失望了，包装破损，客服也不回复",
        "老师讲得很清楚，代码能跑通",
        "系统经常报错，使用体验很差",
        "价格合理，功能实用，整体满意",
        "文档太乱了，安装过程一直失败"
    ]

    result_df = predictor.predict(new_texts)

    print("\n========== 新文本预测结果 ==========")
    print(result_df)


if __name__ == "__main__":
    main()

运行：

python day18_load_and_predict.py
十一、代码重点解释：保存和加载
1. 保存模型
joblib.dump(self.pipeline, self.model_path)

含义是：

把完整 Pipeline 保存成 joblib 文件。

这里保存的不是单独模型，而是：

TF-IDF 向量化器 + 逻辑回归分类器
2. 加载模型
self.pipeline = joblib.load(self.model_path)

含义是：

从硬盘重新加载训练好的 Pipeline。

加载后可以直接：

self.pipeline.predict(texts)
3. 预测类别
pred_labels = self.pipeline.predict(texts)

输出：

0 或 1

其中：

0：差评
1：好评
4. 预测概率
pred_proba = self.pipeline.predict_proba(texts)

输出类似：

[0.15, 0.85]

表示：

差评概率 15%
好评概率 85%
十二、文件 3：Flask API 部署

Flask 可以用很少代码创建 Web 接口，适合入门理解模型服务化。Flask 官方文档说明它是轻量级 WSGI Web 应用框架，强调快速开始并可扩展到复杂应用。

文件名：

day18_flask_api.py

代码如下：

"""
第 18 天：使用 Flask 部署模型预测接口。

启动方式：
python day18_flask_api.py

请求方式：
POST http://127.0.0.1:5000/predict

请求 JSON：
{
    "text": "这个商品质量很好，物流也很快"
}
"""

import re
import jieba
import joblib

from flask import Flask, request, jsonify


class ChineseTextPreprocessor:
    """
    中文文本预处理类。

    注意：
    需要和训练模型时的 tokenizer 保持一致。
    """

    def __init__(self):
        """
        初始化中文文本预处理器。

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

        :param text: 原始文本
        :return: 清洗后的文本
        """
        text = str(text)
        text = re.sub(r"\s+", "", text)
        return text

    def jieba_tokenizer(self, text: str) -> list:
        """
        jieba 中文分词。

        :param text: 原始文本
        :return: 分词结果
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

            if re.fullmatch(r"[，。！？、,.!?；;：:（）()【】\[\]{}]", word):
                continue

            filtered_words.append(word)

        return filtered_words


class FlaskSentimentAPI:
    """
    Flask 情感分类 API 类。

    :param model_path: 模型文件路径
    :return: 无返回值
    """

    def __init__(self, model_path: str = "sentiment_pipeline.joblib"):
        """
        初始化 Flask API。

        :param model_path: 模型文件路径
        :return: 无返回值
        """
        self.model_path = model_path
        self.model = joblib.load(self.model_path)

        self.app = Flask(__name__)
        self.register_routes()

    def predict_text(self, text: str) -> dict:
        """
        对单条文本进行预测。

        :param text: 中文文本
        :return: 预测结果字典
        """
        pred_label = int(self.model.predict([text])[0])
        pred_proba = self.model.predict_proba([text])[0]

        pred_name = "好评" if pred_label == 1 else "差评"

        result = {
            "text": text,
            "pred_label": pred_label,
            "pred_name": pred_name,
            "negative_probability": float(pred_proba[0]),
            "positive_probability": float(pred_proba[1])
        }

        return result

    def register_routes(self):
        """
        注册 Flask 路由。

        :return: 无返回值
        """

        @self.app.route("/", methods=["GET"])
        def index():
            """
            首页接口。

            :return: JSON 响应
            """
            return jsonify({
                "message": "中文情感分类模型 API 正在运行",
                "usage": "POST /predict with JSON {'text': '你的评论内容'}"
            })

        @self.app.route("/predict", methods=["POST"])
        def predict():
            """
            预测接口。

            :return: JSON 响应
            """
            data = request.get_json()

            if data is None:
                return jsonify({
                    "error": "请求体必须是 JSON 格式"
                }), 400

            text = data.get("text", "")

            if not text:
                return jsonify({
                    "error": "字段 text 不能为空"
                }), 400

            result = self.predict_text(text)

            return jsonify(result)

    def run(self):
        """
        启动 Flask 服务。

        :return: 无返回值
        """
        self.app.run(
            host="127.0.0.1",
            port=5000,
            debug=True
        )


def main():
    """
    主函数。

    :return: 无返回值
    """
    api = FlaskSentimentAPI(
        model_path="sentiment_pipeline.joblib"
    )

    api.run()


if __name__ == "__main__":
    main()

运行：

python day18_flask_api.py

测试请求：

curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"这个商品质量很好，物流也很快\"}"

如果你在 Windows PowerShell 里，也可以用：

Invoke-RestMethod `
  -Uri "http://127.0.0.1:5000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"这个商品质量很好，物流也很快"}'
十三、文件 4：FastAPI API 部署

FastAPI 更适合构建现代 API 服务，它基于 Python 类型提示，并能生成 OpenAPI schema 和交互式接口文档；官方文档的 First Steps 页面也展示了自动文档能力。

文件名：

day18_fastapi_api.py

代码如下：

"""
第 18 天：使用 FastAPI 部署模型预测接口。

启动方式：
uvicorn day18_fastapi_api:app --reload --host 127.0.0.1 --port 8000

请求方式：
POST http://127.0.0.1:8000/predict

请求 JSON：
{
    "text": "这个商品质量很好，物流也很快"
}
"""

import re
import jieba
import joblib

from fastapi import FastAPI
from pydantic import BaseModel


class ChineseTextPreprocessor:
    """
    中文文本预处理类。

    注意：
    需要和训练模型时的 tokenizer 保持一致。
    """

    def __init__(self):
        """
        初始化中文文本预处理器。

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

        :param text: 原始文本
        :return: 清洗后的文本
        """
        text = str(text)
        text = re.sub(r"\s+", "", text)
        return text

    def jieba_tokenizer(self, text: str) -> list:
        """
        jieba 中文分词。

        :param text: 原始文本
        :return: 分词结果
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

            if re.fullmatch(r"[，。！？、,.!?；;：:（）()【】\[\]{}]", word):
                continue

            filtered_words.append(word)

        return filtered_words


class PredictRequest(BaseModel):
    """
    预测请求体。

    :param text: 待预测中文评论
    """
    text: str


class PredictResponse(BaseModel):
    """
    预测响应体。

    :param text: 原始输入文本
    :param pred_label: 预测标签
    :param pred_name: 预测名称
    :param negative_probability: 差评概率
    :param positive_probability: 好评概率
    """
    text: str
    pred_label: int
    pred_name: str
    negative_probability: float
    positive_probability: float


class FastAPISentimentService:
    """
    FastAPI 情感分类服务类。

    :param model_path: 模型文件路径
    :return: 无返回值
    """

    def __init__(self, model_path: str = "sentiment_pipeline.joblib"):
        """
        初始化服务类。

        :param model_path: 模型文件路径
        :return: 无返回值
        """
        self.model_path = model_path
        self.model = joblib.load(self.model_path)

    def predict_text(self, text: str) -> dict:
        """
        对单条文本进行预测。

        :param text: 中文评论文本
        :return: 预测结果字典
        """
        pred_label = int(self.model.predict([text])[0])
        pred_proba = self.model.predict_proba([text])[0]

        pred_name = "好评" if pred_label == 1 else "差评"

        return {
            "text": text,
            "pred_label": pred_label,
            "pred_name": pred_name,
            "negative_probability": float(pred_proba[0]),
            "positive_probability": float(pred_proba[1])
        }


app = FastAPI(
    title="中文情感分类模型 API",
    description="使用 TF-IDF + LogisticRegression 的中文评论情感分类接口",
    version="1.0.0"
)

service = FastAPISentimentService(
    model_path="sentiment_pipeline.joblib"
)


@app.get("/")
def index():
    """
    首页接口。

    :return: 服务状态
    """
    return {
        "message": "中文情感分类模型 API 正在运行",
        "usage": "POST /predict with JSON {'text': '你的评论内容'}"
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    情感分类预测接口。

    :param request: 请求体
    :return: 预测响应
    """
    result = service.predict_text(request.text)
    return result

运行：

uvicorn day18_fastapi_api:app --reload --host 127.0.0.1 --port 8000

测试请求：

curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"这个商品质量很好，物流也很快\"}"

Windows PowerShell：

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"这个商品质量很好，物流也很快"}'

启动 FastAPI 后，也可以在浏览器打开：

http://127.0.0.1:8000/docs

这里会看到自动生成的接口文档。

十四、Flask 和 FastAPI 的区别
对比项	Flask	FastAPI
定位	轻量级 Web 框架	现代 API 框架
入门难度	简单	也比较简单
类型提示	不强制	强依赖类型提示
自动文档	默认没有	默认支持
请求校验	需要自己处理或扩展	Pydantic 自动校验
适合场景	小服务、Web 应用、简单接口	API 服务、模型服务、现代后端
启动方式	python app.py	uvicorn app:app --reload

入门阶段可以这样理解：

Flask 更像一个轻量工具箱；
FastAPI 更像专门为 API 服务设计的现代框架。
十五、模型部署的基本流程

真实项目中的模型部署大致是：

训练模型
    ↓
评估模型
    ↓
保存模型文件
    ↓
编写预测函数
    ↓
封装 API 接口
    ↓
启动 Web 服务
    ↓
其他系统通过 HTTP 请求调用模型

例如：

前端页面输入一条评论
    ↓
发送 POST 请求到 /predict
    ↓
后端加载模型并预测
    ↓
返回好评/差评和概率
    ↓
前端展示结果
十六、今天最容易出错的地方
1. 模型文件不存在

如果你直接运行：

python day18_load_and_predict.py

但还没有运行训练文件，就会报错。

正确顺序：

python day18_train_and_save_model.py
python day18_load_and_predict.py
2. 自定义 tokenizer 找不到

因为 Pipeline 中保存了：

jieba_tokenizer

如果加载模型时找不到这个函数或类，可能报错。

所以加载文件中也要有相同的：

ChineseTextPreprocessor

更规范的做法是把它单独放到一个公共文件里，例如：

text_utils.py

训练、预测、API 都从这个文件导入。

3. 只保存了模型，没有保存向量化器

错误做法：

joblib.dump(classifier, "model.joblib")

这样只保存分类器，没有保存 TF-IDF。

正确做法：

joblib.dump(pipeline, "sentiment_pipeline.joblib")
4. 训练和预测时预处理不一致

训练时如果用了：

分词
停用词
TF-IDF
ngram
max_features

预测时也必须一致。

最简单方式就是：

保存完整 Pipeline。
5. Flask debug 不要用于生产环境

今天代码中：

debug=True

是为了学习方便。

真实部署时不要直接使用 debug 模式。

十七、真实项目中还需要考虑什么？

今天只是入门级部署。

真实项目还要考虑：

模型版本管理；
模型文件路径管理；
日志记录；
异常处理；
接口鉴权；
请求限流；
批量预测；
并发性能；
Docker 部署；
模型监控；
数据漂移监控；
定期重新训练。

例如模型文件可以命名为：

sentiment_pipeline_v1.joblib
sentiment_pipeline_v2.joblib
sentiment_pipeline_20260629.joblib

这样方便回滚和管理。

十八、今日练习
练习 1：保存模型

运行：

python day18_train_and_save_model.py

检查当前文件夹下是否出现：

sentiment_pipeline.joblib
练习 2：加载模型预测

运行：

python day18_load_and_predict.py

观察这些文本的预测结果：

这个商品质量很好，物流也很快
太失望了，包装破损，客服也不回复
练习 3：新增测试评论

在 new_texts 中增加：

"课程内容很详细，老师讲得也很清楚"
"代码一直报错，根本跑不通"
"客服回复很慢，问题没有解决"
"价格合理，质量可靠，值得推荐"

观察预测结果是否符合直觉。

练习 4：启动 Flask API

先运行：

python day18_flask_api.py

再使用 PowerShell 请求：

Invoke-RestMethod `
  -Uri "http://127.0.0.1:5000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"老师讲得很清楚，代码也能跑通"}'
练习 5：启动 FastAPI API

运行：

uvicorn day18_fastapi_api:app --reload --host 127.0.0.1 --port 8000

浏览器打开：

http://127.0.0.1:8000/docs

在页面里测试 /predict 接口。

十九、今日学习检查表
检查项	是否掌握
能理解为什么要保存模型	
能使用 joblib.dump() 保存模型	
能使用 joblib.load() 加载模型	
能理解为什么保存完整 Pipeline	
能加载模型预测新数据	
能写 Flask 预测接口	
能写 FastAPI 预测接口	
能使用 curl 或 PowerShell 测试接口	
能理解训练、保存、加载、部署流程	
能知道部署中常见错误	
二十、今日总结

第 18 天最重要的是这句话：

训练模型只是第一步，把模型保存并封装成可调用服务，才是模型真正落地使用的开始。

今天你应该掌握这条流程：

训练模型
    ↓
保存 Pipeline
    ↓
加载 Pipeline
    ↓
预测新数据
    ↓
封装成 API
    ↓
通过 HTTP 请求调用模型

核心代码是：

joblib.dump(pipeline, "sentiment_pipeline.joblib")
pipeline = joblib.load("sentiment_pipeline.joblib")
pipeline.predict(["这个商品质量很好"])

它背后的含义是：

把训练好的完整流程保存起来；
以后输入新文本时，自动完成分词、向量化和分类预测。