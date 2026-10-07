"""
hdfs_modeling.py

Classification and clustering of HDFS block traces.

Pipeline:
    raw log  →  hdfs_parser.py        →  HDFS_Parsed.csv
    HDFS_Parsed.csv  →  hdfs_preprocessor.py  →  HDFS_Templated.csv
    HDFS_Templated.csv + hdfs_traces_labeled.pkl  →  this script

This script runs two analyses on the same trace data:

  1. CLASSIFICATION (supervised, uses labels)
     - Six classifiers compared via 5-fold CV
     - Best models evaluated on a held-out test set
     - Feature importance reported to explain what drives predictions

  2. CLUSTERING (unsupervised, within the Anomaly class only)
     - KMeans on anomaly traces
     - Each cluster described by feature profile vs. overall
     - Examples printed for interpretation
"""

import warnings
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)


# ---- Config ----
TRACES_PATH    = Path("hdfs_traces_labeled.pkl")
TEMPLATED_PATH = Path("HDFS_Templated.csv")

RANDOM_STATE   = 42
TEST_SIZE      = 0.30
N_CLUSTERS     = 3
TSNE_SAMPLE    = 10_000

# Event IDs whose template strings we want to print in the report.
# (These came from the feature-importance and cluster analyses.)
EVENT_IDS_OF_INTEREST = [0, 8, 9, 10, 25, 31, 39, 49, 51, 54]


# =====================================================================
# Shared loading and feature construction
# =====================================================================

def load_traces() -> pd.DataFrame:
    """Load the labeled HDFS traces (one row per block)."""
    traces_df = pd.read_pickle(TRACES_PATH)
    print(f"loaded {len(traces_df):,} traces from {TRACES_PATH}")
    return traces_df


def load_templated_events() -> tuple[pd.DataFrame, dict[int, str]]:
    """Load the templated HDFS events and build an event_id → template map."""
    df = pd.read_csv(TEMPLATED_PATH, dtype=str)
    df["event_id"] = df["event_id"].astype(int)
    vocab = (
        df.drop_duplicates("event_id")
          .set_index("event_id")["template"]
          .to_dict()
    )
    print(f"loaded {len(df):,} events, vocab size {len(vocab)}")
    return df, vocab


def build_features(traces_df: pd.DataFrame) -> pd.DataFrame:
    """Turn each trace into one row of features.

    Features per trace:
      - length       : number of events
      - n_unique     : number of distinct event types
      - first_event  : event_id of the first event
      - last_event   : event_id of the last event
      - event_<id>   : count of event <id> in the trace (bag-of-events)
      - label        : ground-truth class
    """
    traces_df = traces_df.copy()
    traces_df["length"]      = traces_df["sequence"].apply(len)
    traces_df["n_unique"]    = traces_df["sequence"].apply(lambda s: len(set(s)))
    traces_df["first_event"] = traces_df["sequence"].apply(lambda s: s[0] if len(s) else -1)
    traces_df["last_event"]  = traces_df["sequence"].apply(lambda s: s[-1] if len(s) else -1)

    exploded = traces_df[["blockId", "sequence"]].explode("sequence")
    exploded["sequence"] = exploded["sequence"].astype(int)

    bag = (
        exploded.groupby(["blockId", "sequence"])
                .size()
                .unstack(fill_value=0)
                .rename_axis(columns=None)
                .reset_index()
    )
    bag.columns = ["blockId"] + [f"event_{int(c)}" for c in bag.columns[1:]]

    features = traces_df[
        ["blockId", "length", "n_unique", "first_event", "last_event", "label"]
    ].merge(bag, on="blockId", how="left")

    print(f"feature matrix: {features.shape}")
    return features


def print_event_templates(vocab: dict[int, str]) -> None:
    """Print template strings for the event IDs referenced in the report."""
    print("\n=== Event template lookup ===")
    for eid in EVENT_IDS_OF_INTEREST:
        print(f"  event_{eid}: {vocab.get(eid, 'NOT FOUND')}")


# =====================================================================
# Analysis 1: Classification (supervised)
# =====================================================================

def make_pipeline(model) -> Pipeline:
    """Wrap a classifier with scaling and PCA."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.99)),
        ("model", model),
    ])


def build_model_dict() -> dict:
    """The set of classifiers to compare."""
    return {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "qda":      QuadraticDiscriminantAnalysis(reg_param=0.1),
        "knn":      KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "tree":     DecisionTreeClassifier(
                        max_depth=6, class_weight="balanced",
                        random_state=RANDOM_STATE),
        "forest":   RandomForestClassifier(
                        n_estimators=200, class_weight="balanced",
                        n_jobs=-1, random_state=RANDOM_STATE),
        "gb":       GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def run_classification(features: pd.DataFrame) -> None:
    """Compare classifiers via CV and evaluate the ensemble on a held-out set."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION")
    print("=" * 70)

    le = LabelEncoder()
    y = le.fit_transform(features["label"])
    X = features.drop(columns=["blockId", "label"])
    print(f"X shape: {X.shape}")
    print(f"class balance: {dict(zip(le.classes_, np.bincount(y)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # ---- 5-fold CV on the training split ----
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    print(f"\n{'model':<10} {'F1':>7} {'±std':>7}  {'acc':>7} {'prec':>7} {'rec':>7}")
    for name, m in build_model_dict().items():
        s = cross_validate(
            make_pipeline(m), X_train, y_train, cv=cv,
            scoring=["f1", "accuracy", "precision", "recall"],
            n_jobs=-1,
        )
        print(
            f"{name:<10} {s['test_f1'].mean():>7.4f} {s['test_f1'].std():>7.4f}  "
            f"{s['test_accuracy'].mean():>7.4f} "
            f"{s['test_precision'].mean():>7.4f} "
            f"{s['test_recall'].mean():>7.4f}"
        )

    # ---- Held-out test set with classification reports ----
    print("\nHeld-out test set (per-model report):")
    for name, m in build_model_dict().items():
        pipe = make_pipeline(m)
        pipe.fit(X_train, y_train)
        p = pipe.predict(X_test)
        print(f"\n--- {name} ---")
        print(classification_report(y_test, p, target_names=le.classes_))
        print(confusion_matrix(y_test, p))

    # ---- Feature importance via a raw (non-PCA) forest ----
    forest_raw = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    forest_raw.fit(X_train, y_train)
    imp = pd.Series(
        forest_raw.feature_importances_, index=X_train.columns
    ).nlargest(10)
    print("\nTop 10 features (raw forest, no PCA):")
    print(imp.round(4))

    # After training the best classifier on the full training set
    best_pipe = make_pipeline(build_model_dict()['forest'])
    best_pipe.fit(X, y)                                # fit on all data
    joblib.dump(best_pipe, 'hdfs_model.joblib')
    print("saved hdfs_model.joblib")


# =====================================================================
# Analysis 2: Clustering (unsupervised, within Anomaly class)
# =====================================================================

def run_clustering(features: pd.DataFrame,
                   templated_df: pd.DataFrame,
                   vocab: dict[int, str]) -> None:
    """Cluster the Anomaly traces, describe clusters, and save figures."""
    print("\n" + "=" * 70)
    print("CLUSTERING (within Anomaly class)")
    print("=" * 70)

    anomaly_df = features[features["label"] == "Anomaly"].copy()
    X_anomaly = anomaly_df.drop(columns=["blockId", "label"])
    print(f"anomaly traces: {X_anomaly.shape}")

    scaler = StandardScaler()
    X_anomaly_s = scaler.fit_transform(X_anomaly)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    anomaly_df["cluster"] = kmeans.fit_predict(X_anomaly_s)

    # ---- Figure 1: cluster sizes ----
    anomaly_df["cluster"].value_counts().sort_index().plot.bar()
    plt.title("Cluster sizes (Anomaly traces)")
    plt.xlabel("Cluster")
    plt.ylabel("Number of traces")
    plt.savefig("anomaly_cluster_sizes.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("saved anomaly_cluster_sizes.png")

    # ---- Figure 2: feature profile heatmap ----
    top_features = ["length", "n_unique", "first_event", "last_event",
                    "event_0", "event_10", "event_25", "event_31", "event_51"]
    top_features = [f for f in top_features if f in X_anomaly.columns]
    profile = anomaly_df.groupby("cluster")[top_features].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(profile.T, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xticks(range(len(profile)))
    ax.set_xticklabels([f"Cluster {c}" for c in profile.index])
    plt.colorbar(im, label="Mean value")
    plt.title("Feature profile per cluster (Anomaly traces)")
    plt.savefig("anomaly_cluster_profiles.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("saved anomaly_cluster_profiles.png")

    # ---- Figure 3: t-SNE projection on a subsample ----
    n_sample = min(TSNE_SAMPLE, len(X_anomaly_s))
    idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_anomaly_s), size=n_sample, replace=False
    )
    X_tsne = TSNE(
        n_components=2, random_state=RANDOM_STATE,
        perplexity=30, init="pca",
    ).fit_transform(X_anomaly_s[idx])

    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                c=anomaly_df["cluster"].values[idx],
                cmap="viridis", s=8, alpha=0.6)
    plt.title("t-SNE projection of Anomaly traces")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(label="Cluster")
    plt.savefig("anomaly_cluster_tsne.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("saved anomaly_cluster_tsne.png")

    # ---- Silhouette (subsample) ----
    sil = silhouette_score(X_anomaly_s[idx], anomaly_df["cluster"].values[idx])
    print(f"silhouette (10k sample): {sil:.4f}")

    # ---- Cluster descriptions ----
    overall_mean = X_anomaly.mean()
    with open("anomaly_cluster_descriptions.txt", "w", encoding="utf-8") as f:
        for c in sorted(anomaly_df["cluster"].unique()):
            members = X_anomaly[anomaly_df["cluster"] == c]
            cluster_mean = members.mean()
            diff = (cluster_mean - overall_mean).abs().sort_values(ascending=False)

            header = f"\n--- Cluster {c} (n={len(members)}) ---"
            print(header)
            f.write(header + "\n")

            for feat in diff.head(5).index:
                arrow = "↑" if cluster_mean[feat] > overall_mean[feat] else "↓"
                line = (
                    f"  {arrow} {feat:15s}  "
                    f"cluster={cluster_mean[feat]:8.3f}  "
                    f"overall={overall_mean[feat]:8.3f}"
                )
                print(line)
                f.write(line + "\n")
    print("\nsaved anomaly_cluster_descriptions.txt")

    # ---- Examples: one representative block per cluster ----
    for c in sorted(anomaly_df["cluster"].unique()):
        example_id = anomaly_df[anomaly_df["cluster"] == c]["blockId"].iloc[0]
        print(f"\n=== Cluster {c} — example block {example_id} ===")
        msgs = templated_df[templated_df["blockId"] == example_id]["message"].tolist()
        for m in msgs[:8]:
            print("  ", m)


# =====================================================================
# Entry point
# =====================================================================

def main() -> None:
    traces_df = load_traces()
    templated_df, vocab = load_templated_events()
    features = build_features(traces_df)

    run_classification(features)
    run_clustering(features, templated_df, vocab)

    print_event_templates(vocab)


if __name__ == "__main__":
    main()