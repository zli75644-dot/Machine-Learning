第 20 天是收尾复盘课：不再新增一个复杂算法，而是把 20 天内容整理成知识地图、项目流程、常见误区和后续进阶路线。

已思考 5s
第 20 天：机器学习总复盘与进阶路线

今天是 20 天机器学习入门计划的最后一天。

前 19 天你已经从最基础的机器学习概念，一直学习到完整项目实战。第 20 天的目标不是再学一个新模型，而是把前面的内容系统整理成一张完整知识地图。

你要形成一个整体认识：

机器学习不是只会调用 model.fit()；
而是从问题定义、数据处理、特征工程、模型训练、模型评估、模型选择、模型保存到项目落地的一整套流程。
一、今日学习目标

今天需要完成 8 个任务：

复盘 20 天学习内容；
建立机器学习完整知识地图；
理解监督学习、无监督学习、文本机器学习的关系；
总结常见模型的适用场景；
总结完整机器学习项目流程；
梳理常见错误和排查方法；
制定后续进阶路线；
设计可展示的课程项目或科研小项目。
二、20 天学习内容总览
天数	主题	核心内容
第 1 天	机器学习整体认知	X、y、model、fit、predict
第 2 天	sklearn 工具链	numpy、pandas、matplotlib、sklearn
第 3 天	线性回归	回归任务、w、b、MSE
第 4 天	梯度下降	损失函数、梯度、学习率、epoch
第 5 天	逻辑回归	分类概率、Sigmoid、阈值
第 6 天	分类评估指标	Accuracy、Precision、Recall、F1、混淆矩阵
第 7 天	KNN	距离、K 值、标准化
第 8 天	决策树	if-else 规则、Gini、Entropy、max_depth
第 9 天	随机森林	Bagging、多树投票、特征重要性
第 10 天	GBDT / XGBoost / LightGBM	Boosting、逐步修正错误
第 11 天	数据预处理	缺失值、异常值、编码、标准化
第 12 天	特征工程	特征构造、时间特征、分箱、业务特征
第 13 天	模型选择与交叉验证	K 折交叉验证、GridSearchCV
第 14 天	过拟合与欠拟合	泛化能力、模型复杂度、正则化
第 15 天	聚类	无监督学习、KMeans、DBSCAN、用户分层
第 16 天	PCA 降维	主成分、解释方差、二维可视化
第 17 天	文本机器学习	分词、TF-IDF、朴素贝叶斯、情感分类
第 18 天	模型保存与部署	joblib、Pipeline、Flask、FastAPI
第 19 天	完整项目实战	学生学习风险预测完整流程
第 20 天	总复盘与进阶路线	知识体系、项目路线、后续规划
三、机器学习完整知识地图

机器学习可以分成 6 大模块。

机器学习
│
├── 1. 基础概念
│   ├── 样本
│   ├── 特征 X
│   ├── 标签 y
│   ├── 模型 model
│   ├── 训练 fit
│   └── 预测 predict
│
├── 2. 监督学习
│   ├── 回归
│   │   └── 线性回归
│   └── 分类
│       ├── 逻辑回归
│       ├── KNN
│       ├── 决策树
│       ├── 随机森林
│       └── GBDT
│
├── 3. 无监督学习
│   ├── KMeans 聚类
│   ├── DBSCAN 聚类
│   └── PCA 降维
│
├── 4. 数据处理
│   ├── 缺失值处理
│   ├── 异常值处理
│   ├── 类别编码
│   ├── 标准化
│   └── 归一化
│
├── 5. 特征工程
│   ├── 统计特征
│   ├── 时间特征
│   ├── 比例特征
│   ├── 分箱特征
│   └── 业务特征
│
└── 6. 项目落地
    ├── 交叉验证
    ├── 模型调参
    ├── 模型评估
    ├── 模型保存
    └── API 部署

你现在应该从“会跑代码”逐渐升级到“知道一个项目该怎么做”。

四、监督学习总复盘

监督学习的特点是：

有特征 X；
有标签 y；
模型从 X 和 y 中学习规律；
然后对新 X 预测 y。

监督学习分为两类：

回归任务：预测连续数值；
分类任务：预测类别。
1. 回归任务

回归任务的输出是连续值。

例如：

房价预测；
销量预测；
温度预测；
成绩预测；
物流成本预测。

最基础模型是：

线性回归 Linear Regression

核心公式：

y = wx + b

多特征情况下：

y = w1*x1 + w2*x2 + w3*x3 + ... + b

线性回归重点理解：

w：特征权重；
b：偏置；
MSE：均方误差；
目标：让预测值和真实值尽量接近。
2. 分类任务

分类任务的输出是类别。

例如：

是否挂科；
是否患病；
是否垃圾邮件；
是否复购；
是否高风险学生；
评论是好评还是差评。

常见分类模型：

模型	核心思想	适合场景
逻辑回归	线性分数 + Sigmoid 概率	二分类基线
KNN	看最近 K 个邻居投票	小数据、距离明显
决策树	if-else 规则判断	需要解释规则
随机森林	多棵树投票	稳定表格建模
GBDT	一棵树接一棵树修正错误	表格数据强模型
XGBoost	强化版 GBDT	竞赛和工业项目
LightGBM	高效版 GBDT	大规模表格数据
五、分类评估指标总复盘

分类任务不能只看 Accuracy。

你要重点掌握：

混淆矩阵；
Precision；
Recall；
F1-score。
1. 混淆矩阵
真实情况 / 预测结果	预测为 0	预测为 1
真实为 0	TN	FP
真实为 1	FN	TP

解释：

TP：真实为 1，预测为 1；
TN：真实为 0，预测为 0；
FP：真实为 0，预测为 1；
FN：真实为 1，预测为 0。
2. Accuracy
Accuracy = 预测正确的样本数 / 总样本数

适合类别比较均衡的情况。

3. Precision
Precision = TP / (TP + FP)

关注：

模型预测为正类的样本中，有多少是真的正类。

适合误报成本高的场景：

正常邮件被误判成垃圾邮件；
正常用户被误判成风险用户；
正常内容被误判成违规内容。
4. Recall
Recall = TP / (TP + FN)

关注：

真实正类中，有多少被模型找出来。

适合漏报成本高的场景：

疾病筛查；
欺诈交易检测；
高风险学生识别；
设备故障预警。
5. F1-score
F1-score 是 Precision 和 Recall 的综合指标。

适合 Precision 和 Recall 都重要的场景。

六、无监督学习总复盘

无监督学习的特点是：

只有 X；
没有 y；
模型自己发现数据结构。

你学过两个重要方向：

聚类；
降维。
1. 聚类 Clustering

聚类是自动分组。

常见算法：

KMeans；
DBSCAN。

KMeans 核心思想：

指定 K；
随机初始化 K 个中心点；
样本分配给最近中心；
重新计算中心；
不断迭代直到稳定。

DBSCAN 核心思想：

基于密度聚类；
不需要提前指定 K；
可以识别噪声点。

聚类常用于：

用户分层；
商品分组；
客户画像；
异常用户发现；
文章聚合。
2. PCA 降维

PCA 是主成分分析。

核心思想：

把高维特征重新组合成少数几个主成分；
尽量保留原始数据中的主要变化信息。

PCA 常用于：

二维可视化；
高维数据压缩；
降噪；
减少冗余特征；
聚类前降维。

你要记住：

PCA 不是简单删除特征；
PCA 是生成新的主成分。
七、文本机器学习总复盘

文本机器学习的核心是：

先把文本变成数字，再用机器学习模型处理。

完整流程是：

中文文本
    ↓
文本清洗
    ↓
jieba 分词
    ↓
TF-IDF 向量化
    ↓
分类模型
    ↓
预测结果

你学过：

中文分词；
停用词；
CountVectorizer；
TfidfVectorizer；
MultinomialNB；
LogisticRegression；
Pipeline 封装。

文本分类常见任务：

评论情感分类；
垃圾邮件识别；
新闻分类；
工单分类；
舆情分析；
课程评价分类。
八、数据预处理总复盘

真实项目中，数据通常不是干净的。

常见问题：

缺失值；
异常值；
类别字段；
数值范围差异；
重复数据；
无用字段；
特征泄露。

你要掌握的处理方法：

问题	常见处理
数值缺失	均值、中位数填充
类别缺失	众数填充
异常值	删除、截断、设为缺失
类别变量	One-Hot Encoding、Label Encoding
数值尺度差异	StandardScaler、MinMaxScaler
字段太多	特征选择、PCA
字符串文本	分词、TF-IDF

最重要的原则：

训练集 fit；
测试集 transform。

不要把测试集信息泄露到训练过程里。

九、特征工程总复盘

特征工程是机器学习项目中非常重要的一步。

你要记住：

数据预处理让数据能用；
特征工程让数据更有用。

常见特征工程方法：

统计特征；
时间特征；
比例特征；
分箱特征；
组合特征；
业务特征。

例如用户复购预测：

历史订单数；
累计消费金额；
平均订单金额；
最近一次购买距今天数；
购买类目数量；
包邮订单比例；
优惠使用率。

例如学生风险预测：

出勤率；
作业提交率；
实验完成率；
测验平均分；
学习投入指数；
缺勤迟到风险；
是否低出勤；
是否未完成课程设计。

最重要的原则：

特征只能来自预测时已经知道的信息；
不能使用未来信息。

这就是避免特征泄露。

十、模型选择与调参总复盘

模型选择不要只看一次测试结果。

完整流程是：

划分最终测试集
    ↓
训练集内部做交叉验证
    ↓
比较多个模型
    ↓
选择候选模型
    ↓
GridSearchCV / RandomizedSearchCV 调参
    ↓
最终测试集评估
1. 交叉验证

K 折交叉验证的作用：

减少一次数据划分带来的偶然性；
更稳定地比较模型。

常用代码：

cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

多指标评估：

cross_validate(
    model,
    X_train,
    y_train,
    cv=5,
    scoring={
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    }
)
2. GridSearchCV

适合参数组合较少时使用。

GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1",
    cv=5
)
3. RandomizedSearchCV

适合参数组合很多时使用。

RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="f1",
    cv=5
)
十一、过拟合与欠拟合总复盘

你要能通过训练集和测试集表现判断模型状态。

训练集表现	测试集表现	状态
低	低	欠拟合
高	高	拟合较好
很高	明显较低	过拟合
1. 欠拟合

欠拟合就是模型太简单。

解决方法：

增加有效特征；
换更复杂模型；
减少过强正则化；
提高模型复杂度；
做更好的特征工程。
2. 过拟合

过拟合就是模型太复杂。

解决方法：

降低模型复杂度；
增加正则化；
增加训练数据；
删除噪声特征；
使用交叉验证；
限制决策树深度；
增大 min_samples_leaf；
使用早停。
十二、完整机器学习项目模板

以后你做任何机器学习项目，都可以按这个模板走。

1. 明确问题
   - 是分类、回归、聚类还是文本任务？

2. 明确样本粒度
   - 一行是一个用户、一个订单、一个学生，还是一条评论？

3. 明确特征 X 和标签 y
   - X 是输入条件；
   - y 是标准答案。

4. 数据探索
   - 数据多少行多少列？
   - 有没有缺失值？
   - 类别分布是否均衡？
   - 是否有异常值？

5. 数据预处理
   - 缺失值填充；
   - 异常值处理；
   - 类别编码；
   - 标准化。

6. 特征工程
   - 构造统计特征；
   - 构造时间特征；
   - 构造比例特征；
   - 构造业务特征。

7. 划分数据集
   - 训练集；
   - 验证集或交叉验证；
   - 测试集。

8. 训练候选模型
   - 简单模型；
   - 树模型；
   - 集成模型。

9. 模型评估
   - Accuracy；
   - Precision；
   - Recall；
   - F1；
   - 混淆矩阵。

10. 模型选择与调参
    - cross_validate；
    - GridSearchCV；
    - RandomizedSearchCV。

11. 最终测试集评估
    - 只在最后使用测试集。

12. 模型保存
    - 保存完整 Pipeline。

13. 模型部署
    - Flask；
    - FastAPI；
    - Web API。
十三、常用代码模板汇总
1. 分类模型基础模板
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

X = df[feature_columns]
y = df[label_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
2. 预处理 + 模型 Pipeline 模板
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

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

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ]
)

model.fit(X_train, y_train)
3. 交叉验证模板
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

cv_result = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=True
)
4. GridSearchCV 模板
from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [4, 6, 8],
    "classifier__min_samples_leaf": [1, 2, 4]
}

search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

search.fit(X_train, y_train)

print(search.best_params_)
print(search.best_score_)

best_model = search.best_estimator_
5. 模型保存模板
import joblib

joblib.dump(best_model, "model_pipeline.joblib")

loaded_model = joblib.load("model_pipeline.joblib")

y_pred = loaded_model.predict(X_new)
十四、常见模型选择建议
1. 表格二分类任务

例如：

是否挂科；
是否复购；
是否违约；
是否流失。

推荐尝试顺序：

LogisticRegression
DecisionTree
RandomForest
GradientBoosting
XGBoost / LightGBM
2. 表格回归任务

例如：

预测房价；
预测销量；
预测分数；
预测物流成本。

推荐尝试：

LinearRegression
RandomForestRegressor
GradientBoostingRegressor
XGBoostRegressor
LightGBMRegressor
3. 文本分类任务

例如：

评论情感分类；
工单分类；
新闻分类。

推荐尝试：

jieba + TF-IDF + MultinomialNB
jieba + TF-IDF + LogisticRegression
jieba + TF-IDF + LinearSVM
BERT 类预训练模型
4. 用户分层任务

没有标签时：

KMeans
DBSCAN
PCA + KMeans

有标签时：

可以转成监督学习分类任务。
十五、你现在最应该掌握的核心能力

20 天学完后，不要求你马上精通所有模型。

但你必须掌握 6 个核心能力。

1. 能判断任务类型

看到一个问题，要能判断：

这是分类？
这是回归？
这是聚类？
这是文本分类？
这是降维可视化？
2. 能找到 X 和 y

任何监督学习任务都要先问：

X 是什么？
y 是什么？
一行样本代表什么？
3. 能写出 sklearn 标准流程

也就是：

读取数据
    ↓
划分 X/y
    ↓
train_test_split
    ↓
创建模型
    ↓
fit
    ↓
predict
    ↓
evaluate
4. 能做基本数据预处理

至少掌握：

缺失值填充；
类别变量 One-Hot；
标准化；
Pipeline。
5. 能看懂评估指标

不能只看 Accuracy。

要知道：

Precision 代表什么；
Recall 代表什么；
F1 代表什么；
混淆矩阵怎么看。
6. 能做一个完整小项目

例如：

学生挂科风险预测；
课程评论情感分类；
用户复购预测；
商品销量等级预测；
设备故障风险预测。
十六、后续进阶路线

接下来你可以分 4 条路线继续学习。

路线 1：继续深入传统机器学习

适合目标：

打牢 sklearn；
做表格数据项目；
做课程项目或比赛项目。

学习顺序：

1. 更系统学习 pandas 数据分析
2. 更深入学习特征工程
3. 学习 XGBoost
4. 学习 LightGBM
5. 学习模型解释方法 SHAP
6. 学习类别不平衡处理
7. 学习时间序列建模
8. 学习推荐系统基础

建议项目：

学生成绩预测；
用户流失预测；
订单异常检测；
商品销量预测；
客户价值分层。
路线 2：进入深度学习

适合目标：

继续学习计算机视觉；
学习自然语言处理；
学习神经网络；
为科研和课程教学做准备。

学习顺序：

1. 神经网络基础
2. PyTorch 基础
3. 多层感知机 MLP
4. CNN 卷积神经网络
5. RNN / LSTM
6. Transformer
7. BERT
8. YOLO / 图像检测

你之前已经在图像识别、NLP、K210 部署上有基础，所以这条路线很适合继续推进。

路线 3：机器学习项目部署

适合目标：

把模型做成真正可用的小系统。

学习顺序：

1. joblib 保存模型
2. Flask / FastAPI 接口
3. Streamlit 可视化页面
4. Docker 部署
5. 数据库接入
6. 日志记录
7. 模型版本管理
8. 定时重新训练

建议项目：

学生风险预测 Web 系统；
课程评论情感分析系统；
图片分类上传识别系统；
电商用户分层后台。
路线 4：科研方向

适合目标：

看论文；
做方法改进；
写课程设计或科研项目。

学习顺序：

1. 学会复现 baseline
2. 学会读论文 Method
3. 学会分析模型结构图
4. 学会做消融实验
5. 学会对比实验
6. 学会写实验结果表
7. 学会提出改进点
8. 学会整理成论文或项目报告

结合你正在做的图像识别与 NLP 课程，可以考虑：

小目标检测特征增强；
K210 图像识别部署；
轻量化 CNN 模型比较；
中文文本情感分类；
课程学习行为风险预测；
图像识别可视化教学系统。
十七、推荐的 5 个综合项目
项目 1：学生学习风险预测系统

任务类型：

表格二分类

用到知识：

数据预处理；
特征工程；
随机森林；
交叉验证；
模型保存；
FastAPI 部署。

展示价值：

适合课程教学；
可以和学生答辩、课程过程性评价结合。
项目 2：课程评论情感分析系统

任务类型：

中文文本分类

用到知识：

jieba 分词；
TF-IDF；
朴素贝叶斯；
逻辑回归；
Pipeline；
Flask / FastAPI。

展示价值：

可以分析学生评教文本；
可以做 NLP 课程案例。
项目 3：电商用户复购预测

任务类型：

用户行为分类

用到知识：

特征工程；
时间窗口；
用户聚合特征；
随机森林；
GBDT；
模型评估。

展示价值：

适合讲业务特征工程；
适合练习真实表格建模。
项目 4：用户分层与运营策略分析

任务类型：

无监督聚类

用到知识：

KMeans；
DBSCAN；
PCA；
用户画像；
聚类解释。

展示价值：

适合数据分析项目；
可以做可视化报告。
项目 5：K210 图像识别模型对比实验

任务类型：

计算机视觉 + 边缘部署

用到知识：

模型训练；
模型评估；
混淆矩阵；
模型转换；
K210 部署；
推理效果对比。

展示价值：

和你课程内容高度一致；
适合学生课程设计；
能展示从训练到硬件部署的完整链路。
十八、20 天学习后的自测题

你可以用这些问题检查自己是否真正掌握。

基础概念
1. X 和 y 分别是什么？
2. fit 和 predict 分别是什么意思？
3. 分类和回归有什么区别？
4. 监督学习和无监督学习有什么区别？
模型理解
1. 线性回归的 w 和 b 是什么？
2. 逻辑回归为什么用于分类？
3. KNN 为什么需要标准化？
4. 决策树为什么容易过拟合？
5. 随机森林为什么比单棵树稳定？
6. GBDT 和随机森林有什么区别？
评估指标
1. Accuracy 有什么局限？
2. Precision 和 Recall 有什么区别？
3. 医疗筛查为什么更关注 Recall？
4. 混淆矩阵怎么读？
5. F1-score 什么时候有用？
数据处理
1. 缺失值怎么处理？
2. 类别变量怎么编码？
3. One-Hot Encoding 和 Label Encoding 有什么区别？
4. 标准化和归一化有什么区别？
5. 为什么测试集只能 transform，不能 fit_transform？
项目能力
1. 如何定义一个机器学习问题？
2. 如何判断样本粒度？
3. 如何避免特征泄露？
4. 如何使用交叉验证选择模型？
5. 如何保存完整 Pipeline？
6. 如何把模型做成 API？
十九、最容易忘的 10 个关键点
模型只能处理数字，不能直接处理中文文本或类别字符串。
分类任务输出类别，回归任务输出连续数值。
KNN、KMeans、PCA 通常需要标准化。
决策树、随机森林、GBDT 通常不强制标准化。
训练集 fit_transform，测试集 transform。
Accuracy 不是万能指标，类别不均衡时尤其要看 Precision、Recall、F1。
测试集不能反复参与调参。
特征工程不能使用未来信息。
模型保存时最好保存完整 Pipeline。
完整项目不是只训练模型，还包括数据、特征、评估、调参和部署。
二十、你的下一阶段学习安排

建议你接下来用 4 周巩固。

第 1 周：复盘 sklearn 和表格项目

目标：

把第 1 到第 14 天所有代码重新跑一遍。

重点：

LogisticRegression；
RandomForest；
GradientBoosting；
Pipeline；
GridSearchCV。

完成项目：

学生学习风险预测模型。
第 2 周：强化特征工程

目标：

针对一个数据集构造 20 个以上特征。

重点：

统计特征；
时间特征；
比例特征；
分箱特征；
组合特征。

完成项目：

用户复购预测或学生成绩风险预测。
第 3 周：文本机器学习

目标：

做一个中文文本分类器。

重点：

jieba；
TF-IDF；
朴素贝叶斯；
逻辑回归；
SVM。

完成项目：

课程评价情感分析系统。
第 4 周：模型部署与展示

目标：

把一个模型做成可调用系统。

重点：

joblib；
Flask；
FastAPI；
Streamlit；
接口测试。

完成项目：

学生风险预测 API 或中文评论情感分析 API。
二十一、最终总结

这 20 天你完成的是机器学习入门闭环。

你已经从：

不知道机器学习流程

走到了：

能够理解模型、处理数据、训练模型、评估模型、选择模型、保存模型，并做一个完整项目。

最终要记住这条主线：

问题定义
    ↓
数据准备
    ↓
数据预处理
    ↓
特征工程
    ↓
模型训练
    ↓
模型评估
    ↓
模型选择
    ↓
模型调参
    ↓
模型保存
    ↓
模型部署

机器学习真正的能力不是背会某个算法公式，而是能面对一个实际问题，完整地走完这条流程。

你接下来最值得做的一件事是：