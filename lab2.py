import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer


def main():
    df = pd.read_csv("heart_data.csv")

    df.replace("?", np.nan, inplace=True)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    df = pd.DataFrame(imp_mean.fit_transform(df), columns=df.columns)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    results = []

    for iter in range(10):
        for d in range(10):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y
            )
            model = DecisionTreeClassifier(max_depth=d + 1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append(
                {
                    "Iteration": iter + 1,
                    "Accuracy": accuracy,
                    "Model": "DecisionTree",
                    "Max Depth": d + 1,
                }
            )

    results_df = pd.DataFrame(results)
    print(results_df)
    max_accuracy = results_df[results_df["Accuracy"] == results_df["Accuracy"].max()]
    print("Максимальная точность:")
    print(max_accuracy)


if __name__ == "__main__":
    main()
