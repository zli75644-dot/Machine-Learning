第 19 天会把前面所有知识串成一个完整项目：从问题定义、数据构造、预处理、特征工程、模型训练、评估、调参，到最终保存模型。

已思考 7s
第 19 天：完整机器学习项目实战

前 18 天你已经学完了机器学习项目中的大部分核心环节：

机器学习整体流程
sklearn 工具链
回归模型
分类模型
评估指标
KNN
决策树
随机森林
GBDT
数据预处理
特征工程
交叉验证
过拟合与欠拟合
聚类
PCA
文本分类
模型保存与部署

第 19 天要做的事情是：

把前面学过的知识串起来，完成一个完整机器学习小项目。

今天不再单独讲某一个算法，而是按照真实项目流程来完成。

一、今日学习目标

今天你需要掌握 10 个内容：

如何定义一个机器学习问题；
如何确定特征 X 和标签 y；
如何构造模拟项目数据；
如何做数据预处理；
如何做特征工程；
如何训练多个候选模型；
如何使用交叉验证选择模型；
如何使用 GridSearchCV 调参；
如何评估最终模型；
如何保存完整 Pipeline。
二、今日项目主题

今天做一个和你教学场景比较贴近的项目：

学生学习风险预测项目

目标是：

根据学生的学习行为数据，预测学生是否存在挂科风险。

这是一个二分类任务。

标签设计：

0：低风险
1：高风险
三、项目业务背景

假设你是一门课程的老师。

你希望根据学生平时学习行为，提前判断哪些学生可能存在挂科风险。

你有这些数据：

出勤率
作业提交率
实验完成率
课堂测验平均分
每日学习时长
迟到次数
请假次数
专业
年级
是否参加答疑
是否完成课程设计

你希望模型预测：

学生是否高风险

这个项目的价值是：

提前发现风险学生；
提醒学生补交作业；
安排重点辅导；
提高课程通过率；
辅助老师进行教学管理。
四、完整项目流程

今天的完整流程是：

定义问题
    ↓
构造数据
    ↓
查看数据
    ↓
特征工程
    ↓
划分训练集和测试集
    ↓
构建预处理器
    ↓
训练多个模型
    ↓
交叉验证比较模型
    ↓
选择较优模型
    ↓
GridSearchCV 调参
    ↓
最终测试集评估
    ↓
保存模型
    ↓
预测新学生

这就是一个完整机器学习项目的基本骨架。

五、今日项目字段设计
1. 数值特征
字段名	含义
attendance_rate	出勤率
homework_submit_rate	作业提交率
experiment_finish_rate	实验完成率
quiz_avg_score	课堂测验平均分
daily_study_hours	每日学习时长
late_count	迟到次数
leave_count	请假次数
question_count	课堂提问次数
practice_score	上机练习平均分
project_score	课程设计阶段分
2. 类别特征
字段名	含义
major	专业
grade	年级
has_joined_tutoring	是否参加答疑
has_finished_project	是否完成课程设计
3. 构造的新特征
新特征	含义
learning_engagement_score	学习投入指数
assignment_practice_score	作业实验完成指数
absence_risk_score	缺勤迟到风险
is_low_attendance	是否低出勤
is_low_homework	是否低作业提交
is_low_quiz	是否测验低分
4. 标签
字段名	含义
risk_label	是否高风险
0：低风险
1：高风险
六、今日完整代码

新建文件：

day19_complete_ml_project.py

代码如下：

"""
第 19 天：完整机器学习项目实战。

项目主题：
学生学习风险预测。

项目目标：
根据学生的学习行为数据，预测学生是否存在挂科风险。

完整流程：
1. 构造模拟学生学习数据
2. 查看数据基本信息
3. 构造业务特征
4. 区分特征 X 和标签 y
5. 划分训练集和测试集
6. 构建预处理 Pipeline
7. 构建多个候选模型
8. 使用交叉验证比较模型
9. 使用 GridSearchCV 调参
10. 在最终测试集评估最佳模型
11. 保存完整 Pipeline
12. 加载模型预测新学生
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class StudentLearningDataBuilder:
    """
    学生学习数据构造类。

    :param student_count: 学生数量
    :param random_state: 随机种子
    :return: 无返回值
    """

    def __init__(self, student_count: int = 300, random_state: int = 42):
        """
        初始化学生学习数据构造类。

        :param student_count: 学生数量
        :param random_state: 随机种子
        :return: 无返回值
        """
        self.student_count = student_count
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        self.df = None

    def build_data(self) -> pd.DataFrame:
        """
        构造模拟学生学习数据。

        说明：
        这里的数据是模拟数据，用于学习机器学习完整流程。
        真实项目中应替换为真实学生学习过程数据。

        :return: 学生学习数据 DataFrame
        """
        rows = []

        majors = ["物联网", "软件工程", "人工智能", "大数据"]
        grades = ["大一", "大二", "大三"]

        for index in range(1, self.student_count + 1):
            student_id = f"S{index:04d}"

            major = self.rng.choice(majors)
            grade = self.rng.choice(grades)

            attendance_rate = np.clip(
                self.rng.normal(0.82, 0.14),
                0.35,
                1.0
            )

            homework_submit_rate = np.clip(
                self.rng.normal(0.78, 0.18),
                0.20,
                1.0
            )

            experiment_finish_rate = np.clip(
                self.rng.normal(0.75, 0.20),
                0.10,
                1.0
            )

            quiz_avg_score = np.clip(
                self.rng.normal(72, 14),
                20,
                100
            )

            daily_study_hours = np.clip(
                self.rng.normal(2.2, 1.1),
                0,
                7
            )

            late_count = int(
                np.clip(self.rng.poisson(2), 0, 12)
            )

            leave_count = int(
                np.clip(self.rng.poisson(1), 0, 8)
            )

            question_count = int(
                np.clip(self.rng.poisson(3), 0, 15)
            )

            practice_score = np.clip(
                self.rng.normal(74, 15),
                20,
                100
            )

            project_score = np.clip(
                self.rng.normal(76, 16),
                15,
                100
            )

            has_joined_tutoring = self.rng.choice(
                ["是", "否"],
                p=[0.35, 0.65]
            )

            has_finished_project = self.rng.choice(
                ["是", "否"],
                p=[0.78, 0.22]
            )

            """
            构造风险分数。

            风险分数越高，说明学生越可能是高风险。
            这里为了教学演示，用一些业务规则模拟标签。
            """
            risk_score = 0

            risk_score += (1 - attendance_rate) * 2.0
            risk_score += (1 - homework_submit_rate) * 1.8
            risk_score += (1 - experiment_finish_rate) * 1.6
            risk_score += max(0, 70 - quiz_avg_score) / 30
            risk_score += max(0, 70 - practice_score) / 35
            risk_score += max(0, 70 - project_score) / 35
            risk_score += late_count * 0.08
            risk_score += leave_count * 0.12
            risk_score += max(0, 2 - daily_study_hours) * 0.2

            if has_finished_project == "否":
                risk_score += 0.8

            if has_joined_tutoring == "是":
                risk_score -= 0.25

            risk_score += self.rng.normal(0, 0.25)

            risk_label = 1 if risk_score >= 1.85 else 0

            rows.append({
                "student_id": student_id,
                "major": major,
                "grade": grade,
                "attendance_rate": round(float(attendance_rate), 4),
                "homework_submit_rate": round(float(homework_submit_rate), 4),
                "experiment_finish_rate": round(float(experiment_finish_rate), 4),
                "quiz_avg_score": round(float(quiz_avg_score), 2),
                "daily_study_hours": round(float(daily_study_hours), 2),
                "late_count": late_count,
                "leave_count": leave_count,
                "question_count": question_count,
                "practice_score": round(float(practice_score), 2),
                "project_score": round(float(project_score), 2),
                "has_joined_tutoring": has_joined_tutoring,
                "has_finished_project": has_finished_project,
                "risk_label": risk_label
            })

        self.df = pd.DataFrame(rows)

        """
        故意制造少量缺失值，模拟真实数据质量问题。
        """
        missing_columns = [
            "attendance_rate",
            "homework_submit_rate",
            "quiz_avg_score",
            "major"
        ]

        for column in missing_columns:
            missing_index = self.df.sample(
                frac=0.03,
                random_state=self.random_state
            ).index

            self.df.loc[missing_index, column] = np.nan

        return self.df


class StudentFeatureEngineer:
    """
    学生学习风险特征工程类。

    :return: 无返回值
    """

    def __init__(self):
        """
        初始化特征工程类。

        :return: 无返回值
        """
        pass

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构造业务特征。

        新增特征：
        1. learning_engagement_score
        2. assignment_practice_score
        3. absence_risk_score
        4. is_low_attendance
        5. is_low_homework
        6. is_low_quiz

        :param df: 原始学生数据
        :return: 增加特征后的 DataFrame
        """
        df = df.copy()

        df["learning_engagement_score"] = (
            df["attendance_rate"].fillna(df["attendance_rate"].median()) * 0.4
            + df["daily_study_hours"].fillna(df["daily_study_hours"].median()) / 7 * 0.3
            + df["question_count"].fillna(0) / 15 * 0.3
        )

        df["assignment_practice_score"] = (
            df["homework_submit_rate"].fillna(df["homework_submit_rate"].median()) * 0.4
            + df["experiment_finish_rate"].fillna(df["experiment_finish_rate"].median()) * 0.4
            + df["practice_score"].fillna(df["practice_score"].median()) / 100 * 0.2
        )

        df["absence_risk_score"] = (
            df["late_count"].fillna(0) * 0.6
            + df["leave_count"].fillna(0) * 1.0
        )

        df["is_low_attendance"] = (
            df["attendance_rate"].fillna(df["attendance_rate"].median()) < 0.6
        ).astype(int)

        df["is_low_homework"] = (
            df["homework_submit_rate"].fillna(df["homework_submit_rate"].median()) < 0.6
        ).astype(int)

        df["is_low_quiz"] = (
            df["quiz_avg_score"].fillna(df["quiz_avg_score"].median()) < 60
        ).astype(int)

        return df


class StudentRiskModelProject:
    """
    学生学习风险预测完整项目类。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :param model_path: 模型保存路径
    :return: 无返回值
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        model_path: str = "student_risk_model_pipeline.joblib"
    ):
        """
        初始化完整项目类。

        :param test_size: 测试集比例
        :param random_state: 随机种子
        :param model_path: 模型保存路径
        :return: 无返回值
        """
        self.test_size = test_size
        self.random_state = random_state
        self.model_path = model_path

        self.raw_df = None
        self.feature_df = None

        self.label_column = "risk_label"
        self.id_column = "student_id"

        self.numeric_features = [
            "attendance_rate",
            "homework_submit_rate",
            "experiment_finish_rate",
            "quiz_avg_score",
            "daily_study_hours",
            "late_count",
            "leave_count",
            "question_count",
            "practice_score",
            "project_score",
            "learning_engagement_score",
            "assignment_practice_score",
            "absence_risk_score",
            "is_low_attendance",
            "is_low_homework",
            "is_low_quiz"
        ]

        self.categorical_features = [
            "major",
            "grade",
            "has_joined_tutoring",
            "has_finished_project"
        ]

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.candidate_models = None
        self.cv_result_df = None

        self.best_search = None
        self.best_model = None

    def build_data(self):
        """
        构造模拟数据并完成特征工程。

        :return: 无返回值
        """
        data_builder = StudentLearningDataBuilder(
            student_count=300,
            random_state=self.random_state
        )

        self.raw_df = data_builder.build_data()

        feature_engineer = StudentFeatureEngineer()
        self.feature_df = feature_engineer.build_features(self.raw_df)

    def show_data_info(self):
        """
        展示数据基本信息。

        :return: 无返回值
        """
        print("========== 原始数据示例 ==========")
        print(self.raw_df.head())

        print("\n========== 特征工程后数据示例 ==========")
        print(self.feature_df.head())

        print("\n========== 数据形状 ==========")
        print("原始数据形状：", self.raw_df.shape)
        print("特征工程后数据形状：", self.feature_df.shape)

        print("\n========== 缺失值统计 ==========")
        print(self.feature_df.isnull().sum())

        print("\n========== 标签分布 ==========")
        print(self.feature_df[self.label_column].value_counts())

        risk_rate = self.feature_df[self.label_column].mean()
        print(f"\n高风险学生比例：{risk_rate:.4f}")

    def prepare_features_and_label(self):
        """
        准备特征 X 和标签 y。

        :return: 无返回值
        """
        feature_columns = self.numeric_features + self.categorical_features

        self.X = self.feature_df[feature_columns]
        self.y = self.feature_df[self.label_column]

        print("\n========== 特征列 ==========")
        print(feature_columns)

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
        print("训练集特征形状：", self.X_train.shape)
        print("测试集特征形状：", self.X_test.shape)
        print("训练集标签形状：", self.y_train.shape)
        print("测试集标签形状：", self.y_test.shape)

    def build_preprocessor(self) -> ColumnTransformer:
        """
        构建数据预处理器。

        数值特征：
        1. 中位数填充
        2. 标准化

        类别特征：
        1. 众数填充
        2. One-Hot 编码

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

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features)
            ]
        )

        return preprocessor

    def build_candidate_models(self):
        """
        构建多个候选模型 Pipeline。

        :return: 无返回值
        """
        preprocessor = self.build_preprocessor()

        self.candidate_models = {
            "LogisticRegression": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=self.random_state
                        )
                    )
                ]
            ),
            "DecisionTree": Pipeline(
                steps=[
                    ("preprocessor", self.build_preprocessor()),
                    (
                        "classifier",
                        DecisionTreeClassifier(
                            max_depth=5,
                            random_state=self.random_state
                        )
                    )
                ]
            ),
            "RandomForest": Pipeline(
                steps=[
                    ("preprocessor", self.build_preprocessor()),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=150,
                            max_depth=6,
                            min_samples_leaf=2,
                            random_state=self.random_state
                        )
                    )
                ]
            ),
            "GradientBoosting": Pipeline(
                steps=[
                    ("preprocessor", self.build_preprocessor()),
                    (
                        "classifier",
                        GradientBoostingClassifier(
                            n_estimators=120,
                            learning_rate=0.08,
                            max_depth=3,
                            random_state=self.random_state
                        )
                    )
                ]
            )
        }

    def compare_models_by_cross_validation(self):
        """
        使用交叉验证比较多个候选模型。

        :return: 交叉验证结果 DataFrame
        """
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1"
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.random_state
        )

        results = []

        print("\n========== 候选模型 5 折交叉验证对比 ==========")

        for model_name, model in self.candidate_models.items():
            cv_result = cross_validate(
                estimator=model,
                X=self.X_train,
                y=self.y_train,
                cv=cv,
                scoring=scoring,
                return_train_score=True
            )

            result = {
                "model": model_name,
                "test_accuracy_mean": cv_result["test_accuracy"].mean(),
                "test_precision_mean": cv_result["test_precision"].mean(),
                "test_recall_mean": cv_result["test_recall"].mean(),
                "test_f1_mean": cv_result["test_f1"].mean(),
                "train_accuracy_mean": cv_result["train_accuracy"].mean(),
                "fit_time_mean": cv_result["fit_time"].mean()
            }

            results.append(result)

            print(
                f"{model_name}: "
                f"accuracy={result['test_accuracy_mean']:.4f}, "
                f"precision={result['test_precision_mean']:.4f}, "
                f"recall={result['test_recall_mean']:.4f}, "
                f"f1={result['test_f1_mean']:.4f}"
            )

        self.cv_result_df = pd.DataFrame(results)
        self.cv_result_df = self.cv_result_df.sort_values(
            by="test_f1_mean",
            ascending=False
        )

        print("\n========== 模型排行榜：按 F1-score 排序 ==========")
        print(self.cv_result_df)

        return self.cv_result_df

    def tune_random_forest(self):
        """
        使用 GridSearchCV 对随机森林进行调参。

        这里选择随机森林作为最终候选模型。
        实际项目中也可以根据交叉验证排行榜选择其他模型调参。

        :return: GridSearchCV 搜索对象
        """
        pipeline = Pipeline(
            steps=[
                ("preprocessor", self.build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        random_state=self.random_state
                    )
                )
            ]
        )

        param_grid = {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__max_depth": [4, 6, 8, None],
            "classifier__min_samples_leaf": [1, 2, 4]
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.random_state
        )

        self.best_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

        print("\n========== GridSearchCV：随机森林调参 ==========")
        self.best_search.fit(self.X_train, self.y_train)

        print("最佳参数：", self.best_search.best_params_)
        print(f"最佳交叉验证 F1：{self.best_search.best_score_:.4f}")

        self.best_model = self.best_search.best_estimator_

        return self.best_search

    def evaluate_final_model(self):
        """
        在最终测试集上评估最佳模型。

        :return: 无返回值
        """
        y_pred = self.best_model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        print("\n========== 最终测试集评估 ==========")
        print(f"Accuracy：{accuracy:.4f}")
        print(f"Precision：{precision:.4f}")
        print(f"Recall：{recall:.4f}")
        print(f"F1-score：{f1:.4f}")

        print("\n混淆矩阵：")
        print(confusion_matrix(self.y_test, y_pred))

        print("\n分类报告：")
        print(classification_report(
            self.y_test,
            y_pred,
            target_names=["低风险", "高风险"]
        ))

    def show_feature_importance(self):
        """
        输出随机森林模型的特征重要性。

        注意：
        因为类别特征经过 One-Hot 编码，所以需要取出展开后的特征名。

        :return: 无返回值
        """
        classifier = self.best_model.named_steps["classifier"]
        preprocessor = self.best_model.named_steps["preprocessor"]

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

        if not hasattr(classifier, "feature_importances_"):
            print("\n当前模型不支持 feature_importances_。")
            return

        importance_df = pd.DataFrame({
            "feature": all_feature_names,
            "importance": classifier.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="importance",
            ascending=False
        )

        print("\n========== 特征重要性 Top 20 ==========")
        print(importance_df.head(20))

    def save_model(self):
        """
        保存完整 Pipeline。

        :return: 无返回值
        """
        joblib.dump(self.best_model, self.model_path)

        print("\n========== 模型保存完成 ==========")
        print("模型文件路径：", os.path.abspath(self.model_path))

    def predict_new_students(self):
        """
        使用最佳模型预测新学生样本。

        :return: 无返回值
        """
        new_students = pd.DataFrame([
            {
                "major": "物联网",
                "grade": "大二",
                "attendance_rate": 0.55,
                "homework_submit_rate": 0.48,
                "experiment_finish_rate": 0.50,
                "quiz_avg_score": 55,
                "daily_study_hours": 0.8,
                "late_count": 6,
                "leave_count": 3,
                "question_count": 0,
                "practice_score": 52,
                "project_score": 50,
                "has_joined_tutoring": "否",
                "has_finished_project": "否"
            },
            {
                "major": "人工智能",
                "grade": "大三",
                "attendance_rate": 0.92,
                "homework_submit_rate": 0.95,
                "experiment_finish_rate": 0.90,
                "quiz_avg_score": 86,
                "daily_study_hours": 3.2,
                "late_count": 0,
                "leave_count": 0,
                "question_count": 6,
                "practice_score": 88,
                "project_score": 90,
                "has_joined_tutoring": "是",
                "has_finished_project": "是"
            }
        ])

        feature_engineer = StudentFeatureEngineer()
        new_students = feature_engineer.build_features(new_students)

        feature_columns = self.numeric_features + self.categorical_features
        new_X = new_students[feature_columns]

        pred_labels = self.best_model.predict(new_X)
        pred_proba = self.best_model.predict_proba(new_X)

        result_df = new_students.copy()
        result_df["pred_label"] = pred_labels
        result_df["pred_name"] = [
            "高风险" if label == 1 else "低风险"
            for label in pred_labels
        ]
        result_df["low_risk_probability"] = pred_proba[:, 0]
        result_df["high_risk_probability"] = pred_proba[:, 1]

        print("\n========== 新学生预测结果 ==========")
        print(result_df[
            [
                "major",
                "grade",
                "attendance_rate",
                "homework_submit_rate",
                "experiment_finish_rate",
                "quiz_avg_score",
                "has_finished_project",
                "pred_name",
                "low_risk_probability",
                "high_risk_probability"
            ]
        ])

    def load_saved_model_and_predict(self):
        """
        加载保存后的模型，并再次预测新学生。

        这一步验证模型保存和加载是否正常。

        :return: 无返回值
        """
        loaded_model = joblib.load(self.model_path)

        sample = pd.DataFrame([
            {
                "major": "软件工程",
                "grade": "大一",
                "attendance_rate": 0.62,
                "homework_submit_rate": 0.58,
                "experiment_finish_rate": 0.60,
                "quiz_avg_score": 59,
                "daily_study_hours": 1.1,
                "late_count": 4,
                "leave_count": 2,
                "question_count": 1,
                "practice_score": 60,
                "project_score": 58,
                "has_joined_tutoring": "否",
                "has_finished_project": "否"
            }
        ])

        feature_engineer = StudentFeatureEngineer()
        sample = feature_engineer.build_features(sample)

        feature_columns = self.numeric_features + self.categorical_features
        sample_X = sample[feature_columns]

        pred_label = int(loaded_model.predict(sample_X)[0])
        pred_proba = loaded_model.predict_proba(sample_X)[0]

        print("\n========== 加载模型后预测单个学生 ==========")
        print("预测标签：", pred_label)
        print("预测结果：", "高风险" if pred_label == 1 else "低风险")
        print("低风险概率：", round(float(pred_proba[0]), 4))
        print("高风险概率：", round(float(pred_proba[1]), 4))

    def run(self):
        """
        运行完整机器学习项目流程。

        :return: 无返回值
        """
        print("========== 第 19 天：完整机器学习项目实战 ==========")

        self.build_data()
        self.show_data_info()

        self.prepare_features_and_label()
        self.split_data()

        self.build_candidate_models()
        self.compare_models_by_cross_validation()

        self.tune_random_forest()
        self.evaluate_final_model()
        self.show_feature_importance()

        self.save_model()
        self.predict_new_students()
        self.load_saved_model_and_predict()


def main():
    """
    主函数。

    :return: 无返回值
    """
    project = StudentRiskModelProject(
        test_size=0.2,
        random_state=42,
        model_path="student_risk_model_pipeline.joblib"
    )

    project.run()


if __name__ == "__main__":
    main()
七、运行方式

在终端运行：

python day19_complete_ml_project.py

运行成功后，会生成模型文件：

student_risk_model_pipeline.joblib

这个文件保存的是完整 Pipeline，包括：

缺失值填充
标准化
One-Hot 编码
随机森林模型
八、代码重点解释
1. 构造数据
data_builder = StudentLearningDataBuilder(
    student_count=300,
    random_state=42
)
self.raw_df = data_builder.build_data()

这里构造了 300 条学生学习行为数据。

每一行代表一个学生。

2. 特征工程
feature_engineer = StudentFeatureEngineer()
self.feature_df = feature_engineer.build_features(self.raw_df)

这里新增了几个业务特征：

learning_engagement_score
assignment_practice_score
absence_risk_score
is_low_attendance
is_low_homework
is_low_quiz

这些特征不是原始字段，而是根据教学业务理解构造出来的。

例如：

学习投入指数 = 出勤率 + 学习时长 + 提问次数

它比单独看某个字段更能表达学生学习状态。

3. 区分 X 和 y
self.X = self.feature_df[feature_columns]
self.y = self.feature_df[self.label_column]

这里：

X = 学生学习行为特征
y = 是否高风险

这是监督学习标准格式。

4. 构建预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, self.numeric_features),
        ("cat", categorical_transformer, self.categorical_features)
    ]
)

它的作用是：

数值字段：缺失值填充 + 标准化
类别字段：缺失值填充 + One-Hot 编码

这就是第 11 天学习的数据预处理。

5. 构建多个候选模型

代码中比较了 4 个模型：

LogisticRegression
DecisionTree
RandomForest
GradientBoosting

这一步对应第 13 天的模型选择。

6. 使用交叉验证比较模型
cross_validate(
    estimator=model,
    X=self.X_train,
    y=self.y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=True
)

这里会输出多个指标：

accuracy
precision
recall
f1

对于学生风险预测，Recall 很重要。

因为如果学生真实是高风险，但模型没有识别出来，就会漏掉需要帮助的学生。

7. GridSearchCV 调参
param_grid = {
    "classifier__n_estimators": [100, 150, 200],
    "classifier__max_depth": [4, 6, 8, None],
    "classifier__min_samples_leaf": [1, 2, 4]
}

这里对随机森林进行调参。

注意参数名前面有：

classifier__

这是因为模型在 Pipeline 里面，分类器步骤名叫：

"classifier"

所以调参时要写：

classifier__参数名
8. 最终测试集评估
self.best_model.predict(self.X_test)

测试集只在最后使用。

重点看：

Accuracy
Precision
Recall
F1-score
混淆矩阵
分类报告

如果你的目标是尽量发现高风险学生，那么更应该关注：

Recall
9. 特征重要性
classifier.feature_importances_

随机森林可以输出特征重要性。

你可能会看到比较重要的特征：

attendance_rate
homework_submit_rate
experiment_finish_rate
quiz_avg_score
project_score
assignment_practice_score
learning_engagement_score
absence_risk_score

这能帮助你解释：

模型主要根据哪些学习行为判断风险。
10. 保存模型
joblib.dump(self.best_model, self.model_path)

这里保存的是完整 Pipeline。

也就是：

预处理器 + 随机森林模型

以后预测新学生时，不需要重新写标准化和 One-Hot 编码流程。

九、这个项目串联了哪些知识？
知识点	在项目中的体现
监督学习	预测学生是否高风险
分类任务	标签是 0/1
数据预处理	缺失值填充、标准化、One-Hot
特征工程	构造学习投入指数、缺勤风险
模型训练	训练多个分类模型
交叉验证	比较候选模型
评估指标	Accuracy、Precision、Recall、F1
随机森林	最终候选模型
GridSearchCV	超参数调优
过拟合控制	max_depth、min_samples_leaf
模型保存	joblib 保存 Pipeline
模型加载	joblib.load 加载模型
新样本预测	预测新学生是否高风险
十、运行后重点观察什么？
1. 标签分布

你会看到：

risk_label
0    xxx
1    xxx

重点看高风险学生比例。

如果高风险样本太少，那么不能只看 Accuracy。

2. 交叉验证排行榜

重点看：

test_accuracy_mean
test_precision_mean
test_recall_mean
test_f1_mean
train_accuracy_mean

如果训练集分数很高，验证集分数明显低，说明可能过拟合。

3. 最佳参数

GridSearchCV 会输出：

最佳参数
最佳交叉验证 F1

你要理解这些参数为什么可能有效：

max_depth 控制树深度；
min_samples_leaf 控制叶子节点最少样本；
n_estimators 控制树的数量。
4. 混淆矩阵

混淆矩阵可以看出：

低风险学生预测对了多少；
高风险学生预测对了多少；
漏掉了多少高风险学生；
误判了多少低风险学生。

在教学管理场景中：

漏掉高风险学生通常比误判几个低风险学生更严重。

所以 Recall 很关键。

5. 特征重要性

如果模型认为这些特征很重要：

作业提交率
实验完成率
测验平均分
出勤率
课程设计分数

这符合教学直觉。

说明模型学到了一些合理规律。

十一、真实项目中如何改进？

这个项目是模拟数据。真实项目中你可以继续升级：

接入真实学生名单；
接入考勤记录；
接入作业提交记录；
接入实验完成记录；
接入平时测验成绩；
加入课程设计答辩成绩；
加入课堂互动数据；
加入学习平台访问日志。

可以构造更多特征：

近 7 天学习次数
近 30 天作业提交次数
连续缺勤次数
最近一次提交作业距今天数
作业迟交比例
实验代码运行成功次数
课堂测验最低分
成绩波动幅度
答疑参与次数
十二、今日练习
练习 1：修改高风险标签规则

在 StudentLearningDataBuilder 中修改：

risk_label = 1 if risk_score >= 1.85 else 0

分别改成：

risk_label = 1 if risk_score >= 1.6 else 0
risk_label = 1 if risk_score >= 2.1 else 0

观察：

高风险学生比例是否变化；
模型 Accuracy 是否变化；
Recall 是否变化。
练习 2：新增一个特征

新增特征：

score_stability

含义：

测验成绩、练习成绩、课程设计成绩之间的波动程度。

可以写：

df["score_stability"] = df[
    ["quiz_avg_score", "practice_score", "project_score"]
].std(axis=1)

然后加入 numeric_features。

观察模型效果是否变化。

练习 3：关注 Recall

把 GridSearchCV 中：

scoring="f1"

改成：

scoring="recall"

观察最佳参数是否变化。

思考：

如果教学目标是尽量发现高风险学生，为什么 Recall 很重要？
练习 4：替换最终模型

把最终调参模型从随机森林换成 GBDT。

提示：

GradientBoostingClassifier(random_state=self.random_state)

参数搜索可以设置：

param_grid = {
    "classifier__n_estimators": [80, 120, 160],
    "classifier__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "classifier__max_depth": [2, 3, 4]
}

观察效果是否变化。

练习 5：做一个预测接口

结合第 18 天内容，把今天保存的模型：

student_risk_model_pipeline.joblib

封装成 Flask 或 FastAPI 接口。

接口输入：

{
  "major": "物联网",
  "grade": "大二",
  "attendance_rate": 0.55,
  "homework_submit_rate": 0.48,
  "experiment_finish_rate": 0.50,
  "quiz_avg_score": 55,
  "daily_study_hours": 0.8,
  "late_count": 6,
  "leave_count": 3,
  "question_count": 0,
  "practice_score": 52,
  "project_score": 50,
  "has_joined_tutoring": "否",
  "has_finished_project": "否"
}

接口输出：

{
  "pred_name": "高风险",
  "high_risk_probability": 0.87
}
十三、今日学习检查表
检查项	是否掌握
能定义一个机器学习项目问题	
能区分特征 X 和标签 y	
能构造模拟数据	
能完成特征工程	
能使用 ColumnTransformer 做预处理	
能使用 Pipeline 封装流程	
能训练多个候选模型	
能使用交叉验证比较模型	
能使用 GridSearchCV 调参	
能在测试集评估最终模型	
能查看特征重要性	
能保存完整 Pipeline	
能加载模型预测新样本	
十四、今日总结

第 19 天最重要的是这句话：

一个完整机器学习项目，不是只训练一个模型，而是从问题定义、数据处理、特征工程、模型选择、评估调参到模型保存的完整流程。

今天你应该掌握这个项目骨架：

问题定义
    ↓
数据准备
    ↓
特征工程
    ↓
数据预处理
    ↓
模型训练
    ↓
交叉验证
    ↓
模型调参
    ↓
最终评估
    ↓
模型保存
    ↓
新样本预测

你现在应该能看懂下面这类完整 Pipeline：

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier())
    ]
)

它背后的含义是：

先对原始数据做缺失值填充、标准化和 One-Hot 编码；
再把处理后的特征交给随机森林模型训练和预测。