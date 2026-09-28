"""
Mô hình hóa Naive Bayes kết hợp (Mixed / Hybrid Naive Bayes)
Tuân thủ đúng thiết kế:
- age: Gaussian
- balance: Gaussian* (có variance smoothing theo chuẩn Scikit-learn)
- job: Categorical (12 categories, Laplace smoothing alpha=1.0)
- marital: Categorical (3 categories, Laplace smoothing alpha=1.0)
- housing: Bernoulli (yes/no, Laplace smoothing alpha=1.0)
- loan: Bernoulli (yes/no, Laplace smoothing alpha=1.0)
- y: Target (yes/no)
- Giữ nguyên ngưỡng quyết định chuẩn (argmax / threshold = 0.5).
"""

import sys
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class MixedNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    """
    Bộ phân loại Mixed Naive Bayes chuẩn hóa trên Scikit-learn.
    Kết hợp 3 mô hình con: GaussianNB, CategoricalNB, BernoulliNB.
    """
    def __init__(self, alpha=1.0, var_smoothing=1e-9):
        self.alpha = alpha
        self.var_smoothing = var_smoothing
        self.gnb = GaussianNB(var_smoothing=self.var_smoothing)
        self.cnb = CategoricalNB(alpha=self.alpha)
        self.bnb = BernoulliNB(alpha=self.alpha)
        self.encoder = OrdinalEncoder(dtype=int, handle_unknown="use_encoded_value", unknown_value=-1)
        self.classes_ = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        self.classes_ = np.unique(y_train)

        # 1. Tách đặc trưng theo đúng định nghĩa phân phối
        # Gaussian: age, balance
        X_gauss = X_train[["age", "balance"]].values
        
        # Categorical: job, marital (mã hóa sang số nguyên 0, 1, 2, ...)
        X_cat = self.encoder.fit_transform(X_train[["job", "marital"]])
        
        # Bernoulli: housing, loan (mã hóa nhị phân 0 hoặc 1)
        X_bern = (X_train[["housing", "loan"]] == "yes").astype(int).values

        # 2. Huấn luyện từng mô hình thành phần
        self.gnb.fit(X_gauss, y_train)
        self.cnb.fit(X_cat, y_train)
        self.bnb.fit(X_bern, y_train)

        return self

    def _joint_log_likelihood(self, X: pd.DataFrame):
        X_gauss = X[["age", "balance"]].values
        X_cat = self.encoder.transform(X[["job", "marital"]])
        X_bern = (X[["housing", "loan"]] == "yes").astype(int).values

        # Log Prior chung: log P(y)
        log_prior = np.log(self.gnb.class_prior_)

        # Mỗi model._joint_log_likelihood(X) = log P(y) + log P(X_sub | y)
        # Khi cộng 3 mô hình, log P(y) bị lặp lại -> trừ bớt 2 * log P(y)
        jll = (
            self.gnb._joint_log_likelihood(X_gauss)
            + self.cnb._joint_log_likelihood(X_cat)
            + self.bnb._joint_log_likelihood(X_bern)
            - 2 * log_prior
        )
        return jll

    def predict_proba(self, X: pd.DataFrame):
        jll = self._joint_log_likelihood(X)
        return softmax(jll, axis=1)

    def predict(self, X: pd.DataFrame):
        # Quyết định chuẩn Bayes: argmax posterior (ngưỡng 0.5)
        jll = self._joint_log_likelihood(X)
        indices = np.argmax(jll, axis=1)
        return self.classes_[indices]


def run_pipeline():
    print("=" * 70)
    print("DEMO MÔ HÌNH HỌC MÁY MIXED NAIVE BAYES (CHUẨN SCIKIT-LEARN)")
    print("=" * 70)

    # 1. Đọc dữ liệu tập Train và Test
    train_path = "data/train/train.csv"
    test_path = "data/test/test.csv"
    print(f"\n1. Đọc dữ liệu:\n   - Train: {train_path}\n   - Test:  {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = ["age", "balance", "job", "marital", "housing", "loan"]
    target_col = "y"

    X_train = train_df[feature_cols]
    y_train = (train_df[target_col] == "yes").astype(int).values

    X_test = test_df[feature_cols]
    y_test = (test_df[target_col] == "yes").astype(int).values

    print(f"   Kích thước Train: {X_train.shape}, Test: {X_test.shape}")

    # 2. Huấn luyện mô hình
    print("\n2. Huấn luyện mô hình Mixed Naive Bayes...")
    model = MixedNaiveBayesClassifier(alpha=1.0)
    model.fit(X_train, y_train)
    print("   Huấn luyện hoàn tất!")

    # 3. Thông số đã học (Model Parameters) phục vụ demo/thuyết trình
    print("\n3. Các tham số mô hình đã học được:")
    print("-" * 50)
    print(f"   - Xác suất tiên nghiệm (Prior):")
    print(f"     + P(y = no):  {model.gnb.class_prior_[0]:.4f}")
    print(f"     + P(y = yes): {model.gnb.class_prior_[1]:.4f}")

    print("\n   - Phân phối Gaussian (age, balance):")
    print(f"     + Lớp 'no':  mean(age)={model.gnb.theta_[0, 0]:.1f}, mean(balance)={model.gnb.theta_[0, 1]:.1f}")
    print(f"     + Lớp 'yes': mean(age)={model.gnb.theta_[1, 0]:.1f}, mean(balance)={model.gnb.theta_[1, 1]:.1f}")

    print("\n   - Phân phối Bernoulli - Tỉ lệ vay vốn (housing, loan):")
    p_bern_0 = np.exp(model.bnb.feature_log_prob_[0])
    p_bern_1 = np.exp(model.bnb.feature_log_prob_[1])
    print(f"     + Lớp 'no':  P(housing=yes)={p_bern_0[0]:.2%}, P(loan=yes)={p_bern_0[1]:.2%}")
    print(f"     + Lớp 'yes': P(housing=yes)={p_bern_1[0]:.2%}, P(loan=yes)={p_bern_1[1]:.2%}")

    # 4. Đánh giá mô hình trên tập Test (200 mẫu) với ngưỡng chuẩn 0.5
    print("\n4. Đánh giá mô hình trên tập Test (200 mẫu) với ngưỡng chuẩn 0.5:")
    print("-" * 50)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"   * Độ chính xác (Accuracy): {acc:.2%}")
    print(f"   * Diện tích dưới đường cong (ROC-AUC): {roc_auc:.4f}")

    print("\n   * Ma trận nhầm lẫn (Confusion Matrix):")
    print(f"     [[TN={cm[0, 0]:3d}  FP={cm[0, 1]:3d}]")
    print(f"      [FN={cm[1, 0]:3d}  TP={cm[1, 1]:3d}]]")

    print("\n   * Báo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=["no (0)", "yes (1)"], digits=4))

    # 5. Demo dự đoán trên 1 mẫu khách hàng mới
    print("\n5. Thử nghiệm dự đoán trên 1 khách hàng mới:")
    print("-" * 50)
    sample_customer = pd.DataFrame([{
        "age": 35,
        "balance": 2500,
        "job": "management",
        "marital": "married",
        "housing": "no",
        "loan": "no"
    }])
    pred_label = model.predict(sample_customer)[0]
    pred_proba = model.predict_proba(sample_customer)[0]
    print(sample_customer.to_string(index=False))
    print(f"\n   -> Xác suất P(no)={pred_proba[0]:.2%}, P(yes)={pred_proba[1]:.2%}")
    print(f"   -> Kết luận dự đoán: {'Đồng ý gửi tiền (yes)' if pred_label == 1 else 'Từ chối (no)'}")

    return model


if __name__ == "__main__":
    run_pipeline()
