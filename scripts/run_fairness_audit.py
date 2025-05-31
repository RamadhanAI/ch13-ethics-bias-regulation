import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def run_fairness_audit(data_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['target', 'sensitive_feature'])
    y = data['target']
    sensitive = data['sensitive_feature']

    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metric_frame = MetricFrame(metrics=selection_rate,
                               y_true=y_test,
                               y_pred=preds,
                               sensitive_features=sensitive_test)

    print("Selection rates by group:")
    print(metric_frame.by_group)

if __name__ == "__main__":
    import sys
    run_fairness_audit(sys.argv[1])
