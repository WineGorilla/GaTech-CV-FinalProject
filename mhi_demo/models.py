from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_model_specs(k_neighbors: int):
    return {
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k_neighbors)),
        "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3.0, gamma="scale")),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42),
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
        ),
        "dt": DecisionTreeClassifier(max_depth=12, random_state=42),
        "gnb": GaussianNB(),
    }


def parse_requested_models(models_arg: str, model_specs: dict):
    requested_models = [name.strip().lower() for name in models_arg.split(",") if name.strip()]
    unknown_models = [name for name in requested_models if name not in model_specs]
    if unknown_models:
        raise ValueError(
            f"Unknown models in --models: {unknown_models}. Allowed: {list(model_specs.keys())}"
        )
    if not requested_models:
        raise ValueError("No models selected. Pass at least one model in --models.")
    return requested_models
