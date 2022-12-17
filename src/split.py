import math

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from Cat_vs_Cont_stats_and_plots import Cont_Cat_stats_plots
from midterm import BruteForce, read_data


class TrainTestModel(BruteForce, read_data):
    """
    For doing training on sql baseball data we have to
    keep local_date=1 for non_shuffle split that way we can
    take all the old games and predict them on the new ones
    """

    def __init__(self, local_date=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_date = local_date

    def cat_transform(self):
        """
        This is method is to transform categorical predictors into
        continuous variables for all machine learning models that
        I will use. (models only take numbers as input) used method
        from BruteForce module to implement this transformation. Transformation
        is a label encoder method.
        """
        for col_name in list(self.df.columns):
            pred_type = self.get_col_type(col_name)
            if pred_type == "categorical":
                self.df[col_name], _, _ = self.cat_pred_bin(self.df[col_name])
        return self.df

    def sort_split(self):
        self.df = self.cat_transform()
        if self.local_date == 1:
            # shuffle needs to be false for this method
            df = self.df
            X, y = df[self.predictors], df[self.response]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            return X_train, X_test, y_train, y_test
        else:
            X, y = self.df[self.predictors], self.df[self.response]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test

    def trainer_tester(self):
        X_train, X_test, y_train, y_test = self.sort_split()
        try:
            model_1 = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ).fit(X_train, y_train)

            model_2 = LogisticRegression(penalty="l2", random_state=42).fit(
                X_train, y_train
            )
            y_pred_model_1, y_pred_model_2 = model_1.predict(X_test), model_2.predict(
                X_test
            )
            rep_1 = classification_report(y_true=y_test, y_pred=y_pred_model_1)
            rep_2 = classification_report(y_true=y_test, y_pred=y_pred_model_2)
            print("GradientBoostClassifier report: \n", rep_1)
            print("LogisticRegression report: \n", rep_2)
            return model_1, model_2
        except ValueError:
            model_1 = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ).fit(X_train, y_train)

            model_2 = LinearRegression().fit(X_train, y_train)
            y_pred_model_1, y_pred_model_2 = model_1.predict(X_test), model_2.predict(
                X_test
            )
            m_1, m_2 = mean_squared_error(y_test, y_pred_model_1), mean_squared_error(
                y_test, y_pred_model_2
            )
            rep_1 = f"Mean Squared Error of GradientBoostingRegressor: {m_1}"
            rep_2 = f"Mean Squared Error of LinearRegressor: {m_2}"
            print(rep_1, "\n", rep_2)
            return model_1, model_2

    def Feature_Importance(self):
        """
        This method produces a dataframe that has feature importance
        for both models LogRegression and GradientBoost. To get the stats
        values for LogRegression I imported a module that I made called
        Cont_Cat_stats_plots.
        """
        # GradientBoost Feature Ranking
        X = self.df[self.predictors]
        gb_boost_clf, _ = self.trainer_tester()
        df_feature_importance = pd.DataFrame(
            gb_boost_clf.feature_importances_,
            index=X.columns,
            columns=["Feature_Importance"],
        ).sort_values("Feature_Importance", ascending=False)
        stats_values, _, _, _, _, _ = Cont_Cat_stats_plots(
            response=self.response, df=self.df, predictors=self.predictors
        ).predictor_plots()

        # LogisticRegression Ranking
        stats_df = pd.DataFrame(stats_values)
        stats_df.columns = ["Feature", "t-val", "p-val"]
        stats_values, _, _, _, _, _ = Cont_Cat_stats_plots(
            response=self.response, df=self.df, predictors=self.predictors
        ).predictor_plots()
        stats_df = pd.DataFrame(stats_values)
        stats_df.columns = ["Feature", "t-val", "p-val"]

        # I had to make the values floats for some reason?
        # so I am making new float values and storing them in df
        new_vals = []
        _ = [new_vals.append(math.log(1 / float(row))) for row in stats_df["p-val"]]
        stats_df["p-val"] = new_vals
        stats_df = stats_df.sort_values("p-val", ascending=False)
        return df_feature_importance, stats_df

    def plot_feature_rank(self):
        gb_features, LogReg_features = self.Feature_Importance()

        # Gradientboost plot
        sns.set(rc={"figure.figsize": (30, 6)})
        sns.set_style("whitegrid")
        ax1 = sns.barplot(
            data=gb_features, x="Feature_Importance", y=gb_features.index, orient="h"
        )
        ax1.axes.set_title("GradientBoost Feature Ranks")
        ax1.tick_params(labelsize=10)
        plt.show()

        # LogReg plot
        ax2 = sns.barplot(
            x=LogReg_features["p-val"].astype(float),
            y=LogReg_features["Feature"],
            orient="h",
        )
        ax2.axes.set_title("LogisticRegression Feature Ranks")
        ax2.set_xlabel("log(1/(p-val)", fontsize=13)
        ax2.tick_params(labelsize=10)
        plt.show()

        return

    def model_evaluation(self):
        """
        Method is meant to plot the ROC curves of both models
        for a comparison.
        """
        X_train, X_test, y_train, y_test = self.sort_split()
        gb_boost_clf, logReg_clf = self.trainer_tester()
        models = [
            {
                "label": "Logistic Regression",
                "model": logReg_clf,
            },
            {
                "label": "Gradient Boosting",
                "model": gb_boost_clf,
            },
        ]
        fig = go.Figure()
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
        # Below for loop iterates through your models list
        for m in models:
            model = m["model"]  # select the model
            # Compute False postive rate, and True positive rate
            fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            # Calculate Area under the curve to display on the plot
            auc_score = roc_auc_score(y_test, model.predict(X_test))
            # Now, plot the computed values
            name = f"{m['model']} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))
        fig.update_layout(
            title="LogisticRegression vs GradientBoost",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain="domain"),
            width=1100,
            height=900,
        )
        fig.show()
        return
