import sys

import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix


class read_data:
    def __init__(self, pathway="/home/bioinfo/Desktop/BDA_602/src/heart.csv"):
        self.pathway = pathway
        self.df = pd.read_csv(self.pathway, sep=",")
        self.columns = self.df.columns

    def dataframe(self):
        return self.df

    def ChangeBinaryToBool(self):

        for col in self.columns:
            unique = np.sort(np.unique(self.df[col]))
            if len(unique) == 2:
                if unique[0] == 0 and unique[1] == 1:
                    self.df[col] = self.df[col].astype(bool)
                elif (
                    unique[0] == "n"
                    or "N"
                    or "No"
                    or "no"
                    and unique[1] == "y"
                    or "Y"
                    or "yes"
                    or "Yes"
                ):
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
        df = self.ChangeBinaryToBool()

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

        return cont_array, cat_array


class Cat_vs_Cont(read_data):
    def __init__(self, categorical, continuous, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.categorical = categorical
        self.continuous = continuous

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
        return

    def contResponse_vs_catPredictor(self):

        hist_data = [self.df[self.continuous]]
        group_labels = [f"{self.df[self.continuous].name}"]  # name of the dataset
        fig = ff.create_distplot(hist_data, group_labels)
        fig.show()
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
        return

    def contResponse_vs_contPredictor(self):

        # fitting linear regression model for continuous response
        y = self.df[self.continuous]
        column = self.df[self.categorical]  # This is just a place holder for continuous
        predictor = statsmodels.api.add_constant(column)
        linear_regression_model = statsmodels.api.OLS(y, predictor)
        linear_regression_model_fitted = linear_regression_model.fit()
        print(linear_regression_model_fitted.summary())

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
        return


def main():

    # Check if Columns are continous or boolean
    # Also returns tuple with continous and categorical variables
    object = read_data()
    continuous, categorical = object.checkColIsContOrCat()

    # plotting categorical response with continous predictor
    for a, b in zip(categorical, continuous):
        # plots = Cat_vs_Cont(a,b).catResponse_vs_contPredictor()
        pass
    # plotting continuous response with categorical predictor
    for a, b in zip(categorical, continuous):
        # plots = Cat_vs_Cont(a,b).contResponse_vs_catPredictor()
        pass

    # Plotting categorical response with categorical predictor
    for a, b in zip(categorical, sorted(categorical, reverse=True)):
        # test = Cat_vs_Cont(a, b).catResponse_vs_catPredictor()
        pass
    # Plotting continuous response with continuous predictor
    for a, b in zip(continuous, sorted(continuous, reverse=True)):
        test = Cat_vs_Cont(a, b).contResponse_vs_contPredictor()
        pass
    print("doing print statement so I can make a pull request")
    return test


if __name__ == "__main__":
    sys.exit(main())
