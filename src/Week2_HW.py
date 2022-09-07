import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def summary_stats(arr):

    """

    Function that returns pandas DF from
    numpy statistics
    """

    mean = np.array(arr).mean()
    numpy_max = np.max(np.array(arr))
    numpy_min = np.min(np.array(arr))
    quant = np.quantile(np.array(arr), [0.25, 0.5, 0.75])

    # Creating report summary from numpy stats
    # Need to wrap dict in a list to give DF an index
    report = pd.DataFrame(
        [
            {
                "Mean": mean,
                "Max": numpy_max,
                "Min": numpy_min,
                "Quantiles (25%, 50%, 75%)": quant,
            }
        ]
    )
    return report


def main():

    # loading iris dataset from sklearn
    iris = load_iris()

    # Reading in Iris dataset into pandas DF
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df["target"] = pd.Series(iris.target)

    # Summary Statistics
    sepal_length_stats = summary_stats(iris_df["sepal length (cm)"])
    sepal_width_stats = summary_stats(iris_df["sepal width (cm)"])
    petal_length_stats = summary_stats(iris_df["petal length (cm)"])
    petal_width_stats = summary_stats(iris_df["petal width (cm)"])
    target_stats = summary_stats(iris_df["target"])
    print(
        "sepal length summary: \n",
        sepal_length_stats,
        "\nsepeal width summary: \n",
        sepal_width_stats,
        "\npetal length summary: \n",
        petal_length_stats,
        "\npetal width summary: \n",
        petal_width_stats,
        "\ntarget summary: \n",
        target_stats,
    )

    # Scatter Plot
    sctplot = px.scatter(
        data_frame=iris_df, x="sepal length (cm)", y="petal length (cm)", color="target"
    )
    sctplot.show()

    # Boxplot
    boxplt = px.box(
        iris_df,
        x="target",
        y="petal length (cm)",
        color="target",
        notched=True,  # used notched shape
        title="Types of Flowers",
    )
    boxplt.show()

    # Violin plot
    violplt = px.violin(
        iris_df,
        x="target",
        y="petal length (cm)",
        color="target",
        title="Types of Flowers",
    )
    violplt.show()

    # Pair plots
    pairs = px.scatter_matrix(
        iris_df,
        color="target",
        title="Scatter matrix of iris data set",
        labels=iris_df.columns,
    )
    pairs.update_traces(diagonal_visible=False)
    pairs.show()

    # Heatmap
    heat = px.imshow(iris_df)
    heat.show()

    # RandomForest Classifier through pipeline
    print("Model via Pipeline Predictions")
    X_orig = iris_df.loc[:, iris_df.columns != "target"]
    y = iris_df["target"]
    pipeline = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )
    pipeline.fit(X_orig, y)
    probability = pipeline.predict_proba(X_orig)
    prediction = pipeline.predict(X_orig)
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")

    # Gradient Boosted Methods:
    # os.path.dirname(os.path.realpath(__file__))
    # this is a command that lets me find path of current file
    BoostPipe = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=100)),
        ]
    )
    BoostPipe.fit(X_orig, y)
    probability = BoostPipe.predict_proba(X_orig)
    prediction = BoostPipe.predict(X_orig)
    return


if __name__ == "__main__":
    sys.exit(main())
