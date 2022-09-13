import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


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

    # loading iris dataset from UCI
    iris_df = pd.read_csv(
        os.path.dirname(os.path.realpath(__file__)) + "/Iris.csv", sep=","
    )

    # Summary Statistics
    sepal_length_stats = summary_stats(iris_df["SepalLengthCm"])
    sepal_width_stats = summary_stats(iris_df["SepalWidthCm"])
    petal_length_stats = summary_stats(iris_df["PetalLengthCm"])
    petal_width_stats = summary_stats(iris_df["PetalWidthCm"])

    print(
        "sepal length summary: \n",
        sepal_length_stats,
        "\nsepeal width summary: \n",
        sepal_width_stats,
        "\npetal length summary: \n",
        petal_length_stats,
        "\npetal width summary: \n",
        petal_width_stats,
    )

    # Scatter Plot
    sctplot = px.scatter(
        data_frame=iris_df, x="SepalLengthCm", y="PetalLengthCm", color="Species"
    )
    sctplot.show()

    # Boxplot
    boxplt = px.box(
        iris_df,
        x="Species",
        y="PetalLengthCm",
        color="Species",
        notched=True,  # used notched shape
        title="Types of Flowers",
    )
    boxplt.show()

    # Violin plot
    violplt = px.violin(
        iris_df,
        x="Species",
        y="PetalLengthCm",
        color="Species",
        title="Types of Flowers",
    )
    violplt.show()

    # Pair plots
    pairs = px.scatter_matrix(
        iris_df,
        color="Species",
        title="Scatter matrix of iris data set",
        labels=iris_df.columns,
    )
    pairs.update_traces(diagonal_visible=False)
    pairs.show()

    # Heatmap
    heat = px.imshow(iris_df)
    heat.show()

    # Create Encoded Labels
    le = LabelEncoder()

    # Running Data through Classifier
    print("Model via Pipeline Predictions")
    X_orig, y = Data4Pipe(iris_df, le)

    pipeline = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("OneHotEncoder", OneHotEncoder()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )
    pipeline.fit(X_orig, y)
    Metrics(X_orig, pipeline)

    # Gradient Boosted Methods:
    # os.path.dirname(os.path.realpath(__file__))
    # this is a command that lets me find path of current file
    print("BOOOSSSST")

    BoostPipe = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=100)),
        ]
    )
    BoostPipe.fit(X_orig, y)
    Metrics(X_orig, BoostPipe)

    return


def Metrics(X_orig, pipeline):
    probability = pipeline.predict_proba(X_orig)
    prediction = pipeline.predict(X_orig)
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")


def Data4Pipe(iris_df, le):
    X_orig = iris_df.drop(["Species", "Id"], axis=1)
    y = le.fit_transform(iris_df["Species"])
    return X_orig, y


if __name__ == "__main__":
    sys.exit(main())
