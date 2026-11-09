from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).resolve().parents[1] / "kidney_disease.csv"
TARGET = "classification"
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = ["age", "bp", "sg", "su", "bgr", "bu", "sc", "hemo", "pcv"]
CATEGORICAL_FEATURES = ["rbc", "htn", "dm"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_NAMES = ["notckd", "ckd"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = raw.copy()

    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for col in text_cols:
        df[col] = (
            df[col]
            .astype("string")
            .str.replace("\t", "", regex=False)
            .str.strip()
        )
        df[col] = df[col].replace({"": pd.NA, "?": pd.NA, "nan": pd.NA})

    for col in ["pcv", "wc", "rc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

    df[TARGET] = df[TARGET].map({"ckd": 1, "notckd": 0}).astype("int64")
    return df[FEATURES + [TARGET]]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_tree_pipeline() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def print_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    target_names: list[str],
    cm_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print(model_name)
    print(f"{'=' * 60}")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_proba):.4f}")

    print(f"\nConfusion matrix (rows=true, cols=pred; order: {target_names}):")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4,
        )
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(target_names)), labels=target_names)
    ax.set_yticks(range(len(target_names)), labels=target_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=12,
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nConfusion matrix image saved to: {cm_path}")


def cross_validate_tree(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> np.ndarray:
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    target_names = ["notckd" if c == 0 else "ckd" for c in label_encoder.classes_]

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    accuracies = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_enc), start=1):
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_train = y_enc[train_idx]
        y_fold_val = y_enc[val_idx]

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_tree_pipeline()),
                ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        )
        pipeline.fit(X_fold_train, y_fold_train)
        y_fold_pred = pipeline.predict(X_fold_val)

        accuracies.append(accuracy_score(y_fold_val, y_fold_pred))

        print(f"\n{'=' * 60}")
        print(f"Decision Tree — Fold {fold} / {n_splits}")
        print(f"{'=' * 60}")
        print(
            classification_report(
                y_fold_val,
                y_fold_pred,
                target_names=target_names,
                digits=4,
            )
        )

    scores = np.array(accuracies)
    print(f"\n{'=' * 60}")
    print(f"Decision Tree — {n_splits}-Fold Summary")
    print(f"{'=' * 60}")
    for fold, score in enumerate(scores, start=1):
        print(f"Fold {fold} accuracy: {score:.4f}")
    print(f"\nScores       : {np.round(scores, 4)}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Std deviation: {scores.std():.4f}")

    return scores


def explain_logistic(
    preprocessor: ColumnTransformer,
    model: LogisticRegression,
) -> pd.DataFrame:
    feature_names = preprocessor.get_feature_names_out()
    coefs = model.coef_[0]

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefs,
            "Abs_Coefficient": np.abs(coefs),
        }
    ).sort_values(by="Abs_Coefficient", ascending=False)

    print("\nTop 10 Most Predictive Features — Logistic Regression:")
    print(importance_df.head(10).to_string(index=False))
    return importance_df


def explain_tree(
    preprocessor: ColumnTransformer,
    model: DecisionTreeClassifier,
) -> pd.DataFrame:
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    ).sort_values(by="Importance", ascending=False)

    print("\nTop 10 Feature Importances — Decision Tree:")
    print(importance_df.head(10).to_string(index=False))
    return importance_df


def main() -> None:
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Features ({len(FEATURES)}): {FEATURES}")
    print(f"Numeric ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
    print(f"Categorical ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
    print(f"\nTrain shape: {X_train.shape} | Test shape: {X_test.shape}")

    base_dir = Path(__file__).resolve().parent

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(X_train_t, y_train)

    log_pred = log_reg.predict(X_test_t)
    log_proba = log_reg.predict_proba(X_test_t)[:, 1]

    print(
        f"Logistic transformed train shape: {X_train_t.shape} | "
        f"test shape: {X_test_t.shape}"
    )
    print_metrics(
        "Logistic Regression v1",
        y_test,
        log_pred,
        log_proba,
        TARGET_NAMES,
        base_dir / "confusion_matrix.png",
    )
    explain_logistic(preprocessor, log_reg)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    tree_target_names = ["notckd" if c == 0 else "ckd" for c in label_encoder.classes_]

    tree_pipeline = build_tree_pipeline()
    X_train_tree = tree_pipeline.fit_transform(X_train)
    X_test_tree = tree_pipeline.transform(X_test)

    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    tree.fit(X_train_tree, y_train_enc)

    tree_pred = tree.predict(X_test_tree)
    tree_proba = tree.predict_proba(X_test_tree)[:, 1]

    print(
        f"Tree transformed train shape: {X_train_tree.shape} | "
        f"test shape: {X_test_tree.shape}"
    )
    print_metrics(
        "Decision Tree v1",
        y_test_enc,
        tree_pred,
        tree_proba,
        tree_target_names,
        base_dir / "tree_confusion_matrix.png",
    )
    explain_tree(tree_pipeline, tree)

    cross_validate_tree(X, y, n_splits=5)


if __name__ == "__main__":
    main()
