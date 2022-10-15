import sys

import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class read_data:
    def __init__(
        self,
        pathway="/home/bioinfo/Desktop/BDA_602/src/heart.csv",
        response="HeartDisease",
    ):
        self.pathway = pathway
        self.response = response
        self.df = pd.read_csv(self.pathway, sep=",")
        self.columns = self.df.columns

    def dataframe(self):
        return self.df

    def get_response(self):
        return self.response

    def ChangeBinaryToBool(self):

        for col in self.columns:
            unique = np.sort(np.unique(self.df[col]))
            if len(unique) == 2:
                if unique[0] == 0 and unique[1] == 1:
                    self.df[col] = self.df[col].astype(bool)
                elif unique[0].lower() == "n" and unique[1].lower() == "y":
                    # self.df[col] = self.df[col].upper()
                    self.df[col] = self.df[col].map({"Y": 1, "N": 0})
                    self.df[col] = self.df[col].astype(bool)
                else:
                    pass

        bool_df = self.df

        return bool_df

    def checkColIsContOrCat(self):
        """
        Function is meant to determine if columns are categorical or continuous and
        then separate the column names into their respective groups
        """
        # Making arrays to contain column names for each group
        cont_array, cat_array, bool_array = [], [], []

        # Make sure that binary response column is changed into a boolean
        df = self.ChangeBinaryToBool()  # -- method introduces bug in my code

        # Looping through each column and determing conditions
        # to put columns into respective groups
        for index, column in enumerate(df.columns):
            data_type = df[column]
            if data_type.dtype != "bool":
                if data_type.dtype == "float64" or data_type.dtype == "int64":
                    cont_array.append(self.columns[index])
                    print(self.columns[index], " Continuous")
                    continue
                elif data_type.dtype == "string" or data_type.dtype == "object":
                    cat_array.append(self.columns[index])
                    print(self.columns[index], " Not Continuous or Bool")
                    continue
                else:  # Raise Error for unknown data types
                    raise TypeError("Unknown Data Type")
            else:
                bool_array.append(self.columns[index])
                print(self.columns[index], " Boolean")
                continue
            # cat_array = cat_array + bool_array

        # seeing what type of variable the predictor is
        if (self.response == item for item in bool_array):
            group_resp = "boolean"
        elif (self.response == item for item in cat_array):
            group_resp = "categorical"
        elif (self.response == item for item in cat_array):
            group_resp == "continuous"
        else:
            raise TypeError("Unknown Input")

        return cont_array, cat_array, bool_array, group_resp


class Cat_vs_Cont(read_data):
    """
    This object is meant to plot all the types of Data such as:
    Categorical Response vs Categorical Predictor
    Categorical Response vs Continuous Predictor ....etc
    It also takes the p-values of each continuous variable
    """

    def __init__(self, continuous, categorical=None, boolean=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.categorical = categorical
        self.continuous = continuous
        self.boolean = boolean

    def catResponse_vs_contPredictor(self):
        fig_2 = go.Figure()
        fig_2.add_trace(
            go.Violin(
                x=self.df[self.categorical],
                y=self.df[self.continuous],
                name=f"{self.df[self.categorical].name} vs {self.df[self.continuous].name}",
                box_visible=True,
                meanline_visible=True,
            )
        )
        fig_2.update_layout(
            title=f"{self.df[self.categorical].name} vs {self.df[self.continuous].name}",
            xaxis_title=f"{self.df[self.categorical].name}",
            yaxis_title=f"{self.df[self.continuous].name}",
        )
        fig_2.show()
        fig_2.write_html(
            file=f"ViolinPlots: {self.df[self.continuous].name}",
            include_plotlyjs="cdn",
        )
        return

    def contResponse_vs_catPredictor(self):

        hist_data = [self.df[self.continuous]]
        group_labels = [f"{self.df[self.continuous].name}"]  # name of the dataset
        fig = ff.create_distplot(hist_data, group_labels)
        fig.show()

        fig.write_html(
            file=f"ViolinPlots pt 2: {self.df[self.continuous].name}",
            include_plotlyjs="cdn",
        )
        return

    def catResponse_vs_catPredictor(self):

        # I am using self.continuous as a place holder for another categorical variaible
        conf_matrix = confusion_matrix(
            self.df[self.categorical], self.df[self.continuous]
        )
        fig_no_relationship = go.Figure(
            data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
        )
        fig_no_relationship.update_layout(
            title="Categorical Predictor by Categorical Response",
            xaxis_title=f"{self.df[self.categorical].name}",
            yaxis_title=f"{self.df[self.continuous].name}",
        )
        fig_no_relationship.show()

        fig_no_relationship.write_html(
            file=f"HeatMap: {self.df[self.continuous].name} vs {self.df[self.categorical].name} ",
            include_plotlyjs="cdn",
        )
        return

    def contResponse_vs_contPredictor(self):

        y = self.df[self.continuous]
        column = self.df[self.categorical]  # This is just a place holder for continuous
        predictor = statsmodels.api.add_constant(column)
        linear_regression_model = statsmodels.api.OLS(y, predictor)
        linear_regression_model_fitted = linear_regression_model.fit()

        # Get the stats
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

        fig = px.scatter(
            x=self.df[self.continuous], y=self.df[self.categorical], trendline="ols"
        )
        fig.update_layout(
            title=f"Variable: {self.df[self.continuous].name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"{self.df[self.continuous].name}",
            yaxis_title=f"{self.df[self.categorical].name}",
        )
        fig.show()
        fig.write_html(
            file=f"Linear_regression: {self.df[self.continuous].name}",
            include_plotlyjs="cdn",
        )
        return self.df[self.continuous].name, t_value, p_value

    def BoolResponse_vs_ContPredictor(self):

        # if response type is boolean do logistic regression
        df = self.ChangeBinaryToBool()
        # df = self.dataframe()
        df[self.boolean] = df[self.boolean].astype(
            int
        )  # this is absolute necessary for plotting logits
        y = df[self.boolean]
        column = df[self.continuous]
        predictor1 = statsmodels.api.add_constant(column)
        logistic_regression_model = statsmodels.api.Logit(y, predictor1)
        logistic_regression_model_fitted = logistic_regression_model.fit()

        # Getting the stats
        t_value = round(logistic_regression_model_fitted.tvalues[0], 6)
        p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[0])

        # plotting (such a bitch to figure out)
        fig = px.scatter(x=df[self.continuous], y=df[self.boolean], trendline="ols")
        fig.update_layout(
            title=f"Variable: {df[self.continuous].name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"{df[self.continuous].name}",
            yaxis_title=f"{df[self.boolean].name}",
        )
        fig.show()

        fig.write_html(
            file=f"Logistic_regression: {df[self.continuous].name}",
            include_plotlyjs="cdn",
        )

        return df[self.continuous].name, t_value, p_value


class RF_importance(read_data):
    """
    here I am going to require user to input list of names for  each category
    instead of individual column names (unlike Cat_vs_Cont class)
        i.e)list of continuous columns, categorical columns, ....etc
    """

    def __init__(self, continuous, categorical=None, boolean=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.continuous = continuous
        self.categorical = categorical
        self.boolean = boolean

    def RF_regressor(self):
        # loading data
        df = self.ChangeBinaryToBool()
        # df[self.boolean] = df[self.boolean].astype(int) # changing booleans into ints

        # Split data into features and response
        X = df[self.continuous]
        y = df[self.response].astype(int)

        # Train/Test/Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # fit regressor model
        RandForest_regressor = RandomForestRegressor(max_depth=4, random_state=42)
        RF_clf = RandForest_regressor.fit(X_train, y_train)
        df_feature_importance = pd.DataFrame(
            RF_clf.feature_importances_, index=X.columns, columns=["Feature_Importance"]
        ).sort_values("Feature_Importance", ascending=False)

        return df_feature_importance


class DiffMeanResponse(read_data):
    def __init__(self, continuous, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.continuous = continuous

    def upper_lower_bin(self):
        df = self.ChangeBinaryToBool()
        cont = df[self.continuous]
        counts, bins = np.histogram(cont)
        results_df = pd.DataFrame([counts, bins])
        return results_df


def main():

    # read in object and dataframe
    object = read_data()

    # split predictor columns into respective groups
    # and response variable category
    continuous, categorical, boolean, response_VarGroup = object.checkColIsContOrCat()
    response = object.get_response()

    # plotting continuous predictors with response
    stats_values = []
    if continuous is not None:
        for cont_pred in continuous:
            if response_VarGroup == "continuous":
                test = Cat_vs_Cont(cont_pred, response).contResponse_vs_contPredictor()
                stats_values.append(test)
            elif response_VarGroup == "categorical":
                test = Cat_vs_Cont(response, cont_pred).contResponse_vs_catPredictor()
            elif response_VarGroup == "boolean":
                test = Cat_vs_Cont(
                    continuous=cont_pred, boolean=response
                ).BoolResponse_vs_ContPredictor()
                stats_values.append(test)
            else:
                raise TypeError("invalid input...by me")
    else:
        pass

    # Plotting categorical predictors with response
    if categorical is not None:
        for cat_pred in categorical:
            if response_VarGroup == "continuous":
                test = Cat_vs_Cont(response, cat_pred).contResponse_vs_catPredictor()
            elif response_VarGroup == "categorical":
                test = Cat_vs_Cont(response, cat_pred).catResponse_vs_catPredictor()
            else:
                pass
    else:
        pass

    # RF regressor, obtaining feature importance
    machineLearning = RF_importance(continuous)
    ml_df = machineLearning.RF_regressor()

    # combining regression statistics with RF feature importance
    tval, pval = [], []
    tval = [tval.append(stats_values[i][1]) for i in range(0, len(stats_values))]
    pval = [pval.append(stats_values[i][2]) for i in range(0, len(stats_values))]

    ml_df["t-value"] = tval
    ml_df["p-value"] = pval
    print(ml_df)

    return


if __name__ == "__main__":
    sys.exit(main())
