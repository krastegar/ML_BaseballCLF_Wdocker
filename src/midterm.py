import itertools
import pathlib
import random
import sys
import warnings
from argparse import ArgumentError
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from scipy.stats import (
    binned_statistic,
    binned_statistic_2d,
    chi2_contingency,
    pearsonr,
)
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class read_data:
    def __init__(self, response, df, predictors=None):
        self.response = response
        self.df = pd.DataFrame(df)
        self.predictors = predictors
        self.columns = self.df.columns

    def dataframe(self):
        return self.df, print(self.df)

    def ChangeBinaryToBool(self):

        for col in self.columns:
            unique = np.sort(np.unique(self.df[col]))
            if len(unique) == 2:
                if unique[0] == 0 and unique[1] == 1:
                    self.df[col] = self.df[col].astype(bool)
                elif unique[0] in ("n", "N", "no", "No") and unique[1] in (
                    "y",
                    "Y",
                    "yes",
                    "Yes",
                ):
                    self.df[col] = self.df[col].map({unique[1]: 1, unique[0]: 0})
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
        cont_array, cat_array, bool_array= [], [], [],

        # Make sure that binary response column is changed into a boolean
        df = self.ChangeBinaryToBool()  # -- method introduces bug in my code

        # Looping through each column and determing conditions
        # to put columns into respective groups
        list_of_columns = list(df.columns)
        for index, column in enumerate(
            list_of_columns
        ):  # df.columns is a list of columns
            data_type = df[column]
            col_name = self.columns[index]
            if data_type.dtype != "bool":
                if data_type.dtype in ["float64", "int64"]:
                    cont_array.append(col_name)
                    # print(col_name, " Continuous")
                    continue
                elif data_type.dtype in ["string", "object", "category"]:
                    cat_array.append(col_name)
                    # print(col_name, " Not Continuous or Bool")
                    continue
                elif data_type.dtype == 'datetime64[ns]': 
                    df[col_name] = df[col_name].astype(int)
                    cont_array.append(col_name)
                else:  # Raise Error for unknown data types
                    raise TypeError(
                        "Error in column classification: Unknown column dtype"
                    )
                    # print(col_name, data_type.dtype)
            else:
                bool_array.append(col_name)
                # print(col_name, " Boolean")
                continue
            # cat_array = cat_array + bool_array

        return cont_array, cat_array, bool_array

    def get_col_type(self, col):
        """
        Creating a function to return the column type
        """
        cont_array, cat_array, bool_array = self.checkColIsContOrCat()
        if col in bool_array:
            group_resp = "boolean"
        elif col in cat_array:
            group_resp = "categorical"
        elif col in cont_array:
            group_resp = "continuous"
        else:
            raise TypeError("Unknown Input")

        return group_resp


class Cat_vs_Cont(read_data):
    """
    This object is meant to plot all the types of Data such as:
    Categorical Response vs Categorical Predictor
    Categorical Response vs Continuous Predictor ....etc
    It also takes the p-values of each continuous variable.
    This Class is mean to be used in a for loop against individual predictors
    and response
    """

    def __init__(
        self, continuous=None, categorical=None, boolean=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.categorical = categorical
        self.continuous = continuous
        self.boolean = boolean

    def contResponse_vs_catPredictor(self):
        fig_2 = go.Figure()
        fig_2.add_trace(
            go.Violin(
                x=self.df[self.categorical],
                y=self.df[self.response],
                name=f"{self.df[self.categorical].name} vs {self.df[self.response].name}",
                box_visible=True,
                meanline_visible=True,
            )
        )
        fig_2.update_layout(
            title=f"{self.df[self.categorical].name} vs {self.df[self.response].name}",
            xaxis_title=f"{self.df[self.categorical].name}",
            yaxis_title=f"{self.df[self.response].name}",
        )
        # fig_2.show()
        fig_2.write_html(
            file=f"../html_plots_and_tables/ViolinPlots_{self.df[self.categorical].name}.html",
            include_plotlyjs="cdn",
        )
        file_string = f"ViolinPlots_{self.df[self.categorical].name}.html"
        return file_string

    def catResponse_vs_contPredictor(self):

        hist_data = [self.df[self.continuous]]
        group_labels = [f"{self.df[self.response]}"]  # name of the dataset
        fig = ff.create_distplot(hist_data, group_labels)
        # fig.show()

        fig.write_html(
            file=f"../html_plots_and_tables/HistData_Predictor_{self.continuous}_Response={self.response}",
            include_plotlyjs="cdn",
        )
        file_string = f"HistData_Predictor_{self.continuous}_Response={self.response}"
        return file_string

    def catResponse_vs_catPredictor(self):

        # I am using self.continuous as a place holder for another categorical variaible
        conf_matrix = confusion_matrix(
            self.df[self.response].astype(str), self.df[self.categorical].astype(str)
        )
        fig_no_relationship = go.Figure(
            data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
        )
        fig_no_relationship.update_layout(
            title="Categorical Predictor by Categorical Response",
            xaxis_title=f"{self.df[self.response].name}",
            yaxis_title=f"{self.df[self.categorical].name}",
        )
        # fig_no_relationship.show()
        """
        fig_no_relationship.write_html(
            file=f"HeatMap: {self.df[self.categorical].name} vs {self.df[self.response].name} ",
            include_plotlyjs="cdn",
        )
        """
        fig_no_relationship.write_html(
            file=f"../html_plots_and_tables/HeatMap_{self.categorical}_vs_{self.response}.html",
            include_plotlyjs="cdn",
        )
        file_string = f"HeatMap_{self.categorical}_vs_{self.response}.html"
        return file_string

    def contResponse_vs_contPredictor(self):

        y = self.df[self.continuous]
        column = self.df[self.response]  # This is just a place holder for continuous
        predictor = statsmodels.api.add_constant(column)
        linear_regression_model = statsmodels.api.OLS(y, predictor)
        linear_regression_model_fitted = linear_regression_model.fit()

        # Get the stats
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

        fig = px.scatter(
            x=self.df[self.continuous], y=self.df[self.response], trendline="ols"
        )
        fig.update_layout(
            title=f"Variable: {self.continuous}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"{self.continuous}",
            yaxis_title=f"{self.response}",
        )
        # fig.show()
        fig.write_html(
            file=f"../html_plots_and_tables/Linear Regression_{self.continuous}.html",
            include_plotlyjs="cdn",
        )
        file_string = f"Linear Regression_{self.continuous}.html"
        return self.df[self.continuous].name, t_value, p_value, file_string

    def BoolResponse_vs_ContPredictor(self):

        # if response type is boolean do logistic regression
        df = self.ChangeBinaryToBool()
        bool_col = df[self.response]
        # bool col is only 1's....logistic regression will break
        # The way I fixed it was using map function in ChangeBinaryToBool
        y = bool_col
        column = df[self.continuous]
        predictor1 = statsmodels.api.add_constant(column)
        logistic_regression_model = statsmodels.api.Logit(y, predictor1)
        logistic_regression_model_fitted = logistic_regression_model.fit()

        # Getting the stats
        t_value = round(logistic_regression_model_fitted.tvalues[0], 6)
        p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[0])

        # plotting (such a bitch to figure out) ---- how the f did i figure this out
        # the problem is with the y variable, it is in True/False, needed to change it to 0/1
        fig = px.scatter(
            x=self.df[self.continuous], y=df[self.response].astype(int), trendline="ols"
        )
        fig.update_layout(
            title=f"Variable: {df[self.continuous].name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"{df[self.continuous].name}",
            yaxis_title=f"{df[self.response].name}",
        )
        # fig.show()

        fig.write_html(
            file=f"../html_plots_and_tables/Logistic_regression_{df[self.continuous].name}.html",
            include_plotlyjs="cdn",
        )
        file_string = f"Logistic_regression_{df[self.continuous].name}.html"
        return df[self.continuous].name, t_value, p_value, file_string


class RF_importance(read_data):
    """
    here I am going to require user to input list of names for  each category
    instead of individual column names (unlike Cat_vs_Cont class)
        i.e)list of continuous columns, categorical columns, ....etc
    """

    def __init__(
        self,
        regressor=1,
        continuous=None,
        categorical=None,
        boolean=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.continuous = continuous
        self.categorical = categorical
        self.boolean = boolean
        self.regressor = regressor

    def RF_rank(self):
        # loading data
        df = self.ChangeBinaryToBool()
        # df[self.boolean] = df[self.boolean].astype(int) # changing booleans into ints

        # Split data into features and response
        X = df[self.predictors]
        y = df[self.response].astype(int)

        # Train/Test/Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # fit regressor model
        if self.regressor == 1:
            RandForest_regressor = RandomForestRegressor(max_depth=4, random_state=42)
            RF_clf = RandForest_regressor.fit(X_train, y_train)
            df_feature_importance = pd.DataFrame(
                RF_clf.feature_importances_,
                index=X.columns,
                columns=["Feature_Importance"],
            ).sort_values("Feature_Importance", ascending=False)
        elif self.regressor == 0:
            RandForest_regressor = RandomForestClassifier(max_depth=4, random_state=42)
            RF_clf = RandForest_regressor.fit(X_train, y_train)
            df_feature_importance = pd.DataFrame(
                RF_clf.feature_importances_,
                index=X.columns,
                columns=["Feature_Importance"],
            ).sort_values("Feature_Importance", ascending=False)

        else:
            raise ArgumentError("Invalid option inputted")

        return df_feature_importance


class DiffMeanResponse(read_data):
    def __init__(self, pred_input=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred = pred_input

    def bin_params(self, bins_edge):
        bin_lower, bin_upper, bin_center = [], [], []
        # have to start at index 1 and then subtract 1 to get index 0
        for edge in range(1, len(bins_edge)):
            lower = bins_edge[edge - 1]
            upper = bins_edge[edge]
            center = (upper + lower) / 2
            bin_lower.append(lower)
            bin_upper.append(upper)
            bin_center.append(center)
        return bin_lower, bin_upper, bin_center

    def squared_diffs(
        self, bin_means, bins_edge, pop_mean_response, population_proportion
    ):

        mean_square_diff, weighted_square_diff = [], []
        for mean in range(len(bin_means)):
            UnWeighted = (1 / len(bins_edge)) * (
                bin_means[mean] - pop_mean_response
            ) ** 2
            weighted = (population_proportion[mean]) * (
                bin_means[mean] - pop_mean_response
            ) ** 2
            mean_square_diff.append(UnWeighted)
            weighted_square_diff.append(weighted)

        return mean_square_diff, weighted_square_diff

    def Mean_Squared_DF(self):
        """
        Mean of Response calculations and plots. This class is to be used in
        a for loop with individual response and columns
        """
        # Reading in df
        df = self.ChangeBinaryToBool()

        # Getting response type
        response_type = self.get_col_type(self.response)
        predictor_type = self.get_col_type(self.pred)

        # Predictor and response Data
        predictor_data, response_data = df[self.pred], df[self.response]

        # --------------FOUND A WAY TO DO MEAN OF RESPONSE ------------------------------
        # This first condition only deals with categorical predictors not categorical response
        if response_type == "boolean":
            if predictor_type == "categorical":
                le = LabelEncoder()
                label_fit = le.fit_transform(predictor_data)
                predictor_data = label_fit
                bin_size = len(np.unique(predictor_data))
                bin_label = np.unique(le.inverse_transform(label_fit))
            elif predictor_type == "boolean":
                bin_size = len(np.unique(predictor_data))
                predictor_data = predictor_data.astype(int)
                bin_label = np.unique(predictor_data)
            elif predictor_type == "continuous":
                bin_size = 9
                bin_label = np.arange(0, 9)
            else:
                raise TypeError("No Category Matches predictor")

            mean_stat = binned_statistic(
                predictor_data, response_data, statistic="mean", bins=bin_size
            )
            bin_count, bins_edge = np.histogram(predictor_data, bins=bin_size)
            population_proportion = bin_count / len(df)
            bin_means = mean_stat.statistic
            pop_mean_response = np.mean(response_data)

            bin_lower, bin_upper, bin_center = self.bin_params(bins_edge)
            mean_square_diff, weighted_diff = self.squared_diffs(
                bins_edge=bins_edge,
                bin_means=bin_means,
                pop_mean_response=pop_mean_response,
                population_proportion=population_proportion,
            )
            # bin_label = np.unique(le.inverse_transform(label_fit))

        elif response_type == "continuous":
            if predictor_type == "categorical":
                le = LabelEncoder()
                label_fit = le.fit_transform(predictor_data)
                predictor_data = label_fit
                bin_size = len(np.unique(predictor_data))
                bin_label = np.unique(le.inverse_transform(label_fit))
            elif predictor_type == "boolean":
                bin_size = len(np.unique(predictor_data))
                predictor_data = predictor_data.astype(int)
                bin_label = np.unique(predictor_data)
            elif predictor_type == "continuous":
                bin_size = 9
                bin_label = np.arange(0, 9)
            else:
                raise TypeError("No Category Matches predictor")

            mean_stat = binned_statistic(
                predictor_data, response_data, statistic="mean", bins=bin_size
            )
            pop_mean_response = np.mean(response_data)
            bin_count, bins_edge = np.histogram(predictor_data, bins=bin_size)
            population_proportion = bin_count / len(df)
            bin_means = mean_stat.statistic
            bin_lower, bin_upper, bin_center = self.bin_params(bins_edge)
            mean_square_diff, weighted_diff = self.squared_diffs(
                bins_edge=bins_edge,
                bin_means=bin_means,
                pop_mean_response=pop_mean_response,
                population_proportion=population_proportion,
            )

        else:
            raise TypeError("Response needs to be continuous or boolean")

        mean_diff_df = pd.DataFrame(
            {
                "Bin": bin_label,
                "LowerBin": bin_lower,
                "UpperBin": bin_upper,
                "BinCenters": bin_center,
                "BinCount": bin_count,
                "BinMeans": bin_means,
                "PopulationMean": pop_mean_response,
                "PopulationProportion": population_proportion,
                "MeanSquaredDiff": mean_square_diff,
                "MeanSquaredDiffWeighted": weighted_diff,
            }
        )
        mean_diff_df = mean_diff_df.sort_values(
            "MeanSquaredDiffWeighted", ascending=False
        ).reset_index(drop=True)
        return mean_diff_df

    def plot_Mean_diff(self):

        mean_diff_df = self.Mean_Squared_DF()
        fig = go.Figure(
            layout=go.Layout(
                title="Combined Layout",
                yaxis2=dict(overlaying="y", side="right"),
            )
        )
        fig.add_trace(
            go.Bar(
                name="counts",
                x=mean_diff_df["BinCenters"],
                y=mean_diff_df["BinCount"],
                yaxis="y1",
            ),
        )
        fig.add_trace(
            go.Scatter(
                name="Pop-Mean",
                x=mean_diff_df["BinCenters"],
                y=mean_diff_df["PopulationMean"],
                yaxis="y2",
            )
        )
        fig.add_trace(
            go.Scatter(
                name="Squared diff",
                x=mean_diff_df["BinCenters"],
                y=mean_diff_df["BinMeans"],
                yaxis="y2",
            )
        )
        fig.update_layout(title=f"Difference w/ Mean Response: {self.pred}")
        fig.update_layout(
            xaxis_title=f"{self.pred}",
            yaxis_title="Population",
            yaxis2_title="Mean of Response",
        )
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(
            tickvals=mean_diff_df["BinCenters"], ticktext=mean_diff_df["Bin"]
        )

        # fig.show()
        fig.write_html(
            file=f"../html_plots_and_tables/Mean_of_Response_response_=_{self.response}_vs_{self.pred}.html",
            include_plotlyjs="cdn",
        )
        file_string = f"Mean_of_Response_response_=_{self.response}_vs_{self.pred}.html"
        return file_string

class Correlation(read_data):
    """
    Determining correlation of PREDICTORS only
    ***Predictors of all categories.
    """

    def __init__(self, a=None, b=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a  # these are column titles...just fyi
        self.b = b

    def get_df(self):

        df = self.ChangeBinaryToBool()
        df_a, df_b = df[self.a], df[self.b]
        return df_a, df_b

    def cont_cont_Corr(self):
        "Remember to set a condition for a for loop....if a==b: pass"
        corr_type = "Continuous vs Continuous"
        df_a, df_b = self.get_df()
        rval, _ = pearsonr(df_a, df_b)  # gets rid of unwanted variable

        return self.a, self.b, rval, corr_type

    def cat_cont_correlation_ratio(self, categories, values):
        """
        Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
        SOURCE:
        1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
        :param categories: Numpy array of categories
        :param values: Numpy array of values
        :return: correlation
        """
        corr_type = "Continuous vs Categorical"
        f_cat, _ = pd.factorize(categories)
        cat_num = np.max(f_cat) + 1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0, cat_num):
            cat_measures = values[(np.argwhere(f_cat == i).flatten())]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
        numerator = np.sum(
            np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
        )
        denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = np.sqrt(numerator / denominator)
        return self.b, self.a, eta, corr_type

    def fill_na(self, data):
        if isinstance(data, pd.Series):
            return data.fillna(0)
        else:
            return np.array([value if value is not None else 0 for value in data])

    def cat_correlation(self, x, y, bias_correction=True, tschuprow=False):
        """
        Calculates correlation statistic for categorical-categorical association.
        The two measures supported are:
        1. Cramer'V ( default )
        2. Tschuprow'T
        SOURCES:
        1.) CODE: https://github.com/MavericksDS/pycorr
        2.) Used logic from:
            https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
            to ignore yates correction factor on 2x2
        3.) Haven't validated Tschuprow
        Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
        Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
        Parameters:
        -----------
        x : list / ndarray / Pandas Series
            A sequence of categorical measurements
        y : list / NumPy ndarray / Pandas Series
            A sequence of categorical measurements
        bias_correction : Boolean, default = True
        tschuprow : Boolean, default = False
                For choosing Tschuprow as measure
        Returns:
        --------
        float in the range of [0,1]
        """
        corr_type = "Cat vs Cat"
        corr_coeff = np.nan
        try:
            x, y = self.fill_na(x), self.fill_na(y)
            crosstab_matrix = pd.crosstab(x, y)
            n_observations = crosstab_matrix.sum().sum()

            yates_correct = True
            if bias_correction:
                if crosstab_matrix.shape == (2, 2):
                    yates_correct = False

            chi2, _, _, _ = chi2_contingency(crosstab_matrix, correction=yates_correct)
            phi2 = chi2 / n_observations

            # r and c are number of categories of x and y
            r, c = crosstab_matrix.shape
            if bias_correction:
                phi2_corrected = max(
                    0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1)
                )
                r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
                c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
                if tschuprow:
                    corr_coeff = np.sqrt(
                        phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                    )
                    return self.a, self.b, corr_coeff, corr_type
                corr_coeff = np.sqrt(
                    phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
                )
                return self.a, self.b, corr_coeff,corr_type
            if tschuprow:
                corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
                return self.a, self.b, corr_coeff, corr_type
            corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
            return self.a, self.b, corr_coeff, corr_type
        except Exception as ex:
            print(ex)
            if tschuprow:
                warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
            else:
                warnings.warn("Error calculating Cramer's V", RuntimeWarning)
            return self.a, self.b, corr_coeff, corr_type

    def cont_vs_cont_matrix(self):
        """
        corr vals is an array of all the correlation values computed in the
        previous correlation anaylsis
        """
        cont_array, _, _ = self.checkColIsContOrCat()
        df = self.ChangeBinaryToBool()
        cont_matrix = df[cont_array].corr(method="pearson")

        fig = px.imshow(cont_matrix)

        fig.update_layout(title="Correlation Matrix: Continuous / Continuous ")
        fig.show()

        fig.write_html(
            file="../html_plots_and_tables/Correlation_Matrix_Continuous_vs_Continuous.html",
            include_plotlyjs="cdn",
        )
        file_string = "Correlation_Matrix_Continuous_vs_Continuous"
        return file_string

    def cat_vs_cont_matrix(self, corr_matrix):
        cat_catDF = corr_matrix.pivot_table(
            values="Corr Coef",
            index="Predictor 1",
            columns="Predictor 2",
            aggfunc="first",
        )
        fig = px.imshow(cat_catDF)

        fig.update_layout(title="Correlation Matrix: Continuous / Categorical ")
        fig.write_html(
            file="../html_plots_and_tables/Correlation Matrix_Continuous_vs_Categorical.html",
            include_plotlyjs="cdn",
        )

        fig.show()
        return

    def cat_vs_cat_matrix(self, corr_matrix):
        cat_catDF = corr_matrix.pivot_table(
            values="Corr Coef",
            index="Predictor 1",
            columns="Predictor 2",
            aggfunc="first",
        )
        fig = px.imshow(cat_catDF)

        fig.update_layout(title="Correlation Matrix: Categorical / Categorical ")
        fig.write_html(
            file="../html_plots_and_tables/Correlation Matrix_Categorical_vs_Categorical.html",
            include_plotlyjs="cdn",
        )
        fig.show()
        file_string = "Correlation Matrix_Categorical_vs_Categorical.html"
        return file_string


class BruteForce(DiffMeanResponse):
    def __init__(self, list1, list2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.list1 = list1
        self.list2 = list2

    def cat_pred_bin(self, predictor_data):

        le = LabelEncoder()
        label_fit = le.fit_transform(predictor_data)
        predictor_data = label_fit
        bin_size = len(np.unique(predictor_data))
        bin_label = np.unique(le.inverse_transform(label_fit))

        return predictor_data, bin_size, bin_label

    def brute_force_2d(self, pred1, pred2):

        # Reading in df
        df = self.ChangeBinaryToBool()

        # Getting response type
        # response_type = self.get_col_type(self.response)

        pred1_type = self.get_col_type(pred1)
        # print("\npredictor 1 type: ", pred1_type)
        # print('\n predictor 2: ', self.pred2)
        pred2_type = self.get_col_type(pred2)
        # print("\npredictor 2 type: ", pred2_type)
        # Predictor and response Data
        pred1_data, pred2_data, response_data = df[pred1], df[pred2], df[self.response]

        # Conditions for predictors
        if pred1_type == "continuous" and pred2_type == "continuous":
            bin_size = 10
            bin_label = np.arange(0, 9)
            brute_stats = binned_statistic_2d(
                pred1_data, pred2_data, response_data, statistic="mean", bins=bin_size
            )
            count_stats = binned_statistic_2d(
                pred1_data, pred2_data, response_data, statistic="count", bins=bin_size
            )
        elif pred1_type in ("categorical", "boolean") and pred2_type == "continuous":
            pred1_data, bin_size, bin_label = self.cat_pred_bin(pred1_data)
            brute_stats = binned_statistic_2d(
                pred1_data, pred2_data, response_data, statistic="mean", bins=bin_size
            )
            count_stats = binned_statistic_2d(
                pred1_data, pred2_data, response_data, statistic="count", bins=bin_size
            )
        elif pred1_type == "continuous" and pred2_type in ("categorical", "boolean"):
            pred2_data, bin_size, bin_label = self.cat_pred_bin(pred2_data)
            brute_stats = binned_statistic_2d(
                pred1_data, pred2_data, response_data, statistic="mean", bins=bin_size
            )
            count_stats = binned_statistic_2d(
                pred1_data, pred2_data, response_data, statistic="count", bins=bin_size
            )

        elif pred1_type == "categorical" and pred2_type == "categorical":
            # need to find a way to choose between bin_sizes
            # just gonna use the larger one of the two
            pred1_data, bin_size, bin_label_1 = self.cat_pred_bin(pred1_data)
            pred2_data, bin_size, bin_label = self.cat_pred_bin(pred2_data)

            brute_stats = binned_statistic_2d(
                pred1_data, pred2_data, response_data, statistic="mean", bins=bin_size
            )
            count_stats = binned_statistic_2d(
                pred1_data, pred2_data, response_data, statistic="count", bins=bin_size
            )
        else:
            raise ArgumentError("Not the correct Predictor Type for BruteForce Method")

        pop_mean_response = np.mean(response_data)

        bin_counts, x_edge, y_edge = np.histogram2d(
            pred1_data, pred2_data, bins=bin_size
        )
        # bin_area = bin_counts.flatten() # this is giving me the total sample size of each bin
        matrix_area = len(x_edge) * len(
            y_edge
        )  # area of entire nxn matrix (based on # of bins)
        bin_means = brute_stats.statistic.flatten()
        bin_means_noNan = np.nan_to_num(bin_means)  # switching nan to 0's
        population_proportion = bin_means_noNan / matrix_area

        # replacing nan with 0 in bin_means

        mean_square_diff, weighted_diff = self.squared_diffs(
            bins_edge=bin_means,
            bin_means=bin_means_noNan,
            pop_mean_response=pop_mean_response,
            population_proportion=population_proportion,
        )
        # -------- plot just using 2d values ---------------
        # new plan : going to use counts df and mean df for population proportion
        # then do calculations on matrices

        bin_2dstats = brute_stats.statistic
        # test_structure = (bin_2dstats, x_edge, y_edge, bin_number)
        bin_2dstats = bin_2dstats
        count_2d = count_stats.statistic
        pop_2dArray = bin_2dstats / count_2d
        weighted_means = pop_2dArray * (bin_2dstats - pop_mean_response) ** 2

        return pred1, pred2, mean_square_diff, weighted_diff, weighted_means

    def BF_df(self):
        cont_cat_bruteforce = []
        for tupl in itertools.product(self.list1, self.list2):
            if tupl[0] == tupl[1]:
                continue
            BruF = self.brute_force_2d(pred1=tupl[0], pred2=tupl[1])
            pred1, pred2, mean_square_diff, weighted_diff, _ = BruF
            mean_unweighted, mean_weighted = np.mean(mean_square_diff), np.mean(
                weighted_diff
            )
            cont_cat_bruteforce.append((pred1, pred2, mean_unweighted, mean_weighted))
        BF_cont_cat_df = pd.DataFrame(
            cont_cat_bruteforce, columns=["Pred 1", "Pred 2", "Unweighted", "Weighted"]
        )

        return BF_cont_cat_df.sort_values("Weighted", ascending=False)

    def plot_func(self, brutforce_df, pred1, pred2, title):
        # title should be either, catVScat, catVScont, contVScont
        fig = px.imshow(brutforce_df)

        fig.update_layout(title=f"BruteForce {title}: {pred1} vs {pred2}")
        fig.write_html(
            file=f"../html_plots_and_tables/BruteForce_{title}_{pred1}_vs_{pred2}.html",
            include_plotlyjs="cdn",
        )
        # fig.show()
        file_string = f"BruteForce_{title}_{pred1}_vs_{pred2}.html"

        return file_string

    def plot_brutforce(self, title):
        file_names = []
        for tupl in itertools.product(self.list1, self.list2):
            if tupl[0] == tupl[1]:
                continue
            brutDF = self.brute_force_2d(pred1=tupl[0], pred2=tupl[1])
            _, _, _, _, bf_df = brutDF
            cont_cat_brutDF = self.plot_func(
                bf_df, pred1=tupl[0], pred2=tupl[1], title=title
            )
            file_names.append(cont_cat_brutDF)
        return file_names


def get_test_data_set(data_set_name: str = None) -> (pd.DataFrame, List[str], str):
    """Function to load a few test data sets
    :param:
    data_set_name : string, optional
        Data set to load
    :return:
    data_set : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.
    predictors :list[str]
        List of predictor variables
    response: str
        Response variable
    """
    seaborn_data_sets = ["mpg", "tips", "titanic", "titanic_2"]
    sklearn_data_sets = ["boston", "diabetes", "breast_cancer"]
    all_data_sets = seaborn_data_sets + sklearn_data_sets

    TITANIC_PREDICTORS = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "embarked",
        "parch",
        "fare",
        "who",
        "adult_male",
        "deck",
        "embark_town",
        "alone",
        "class",
    ]
    if data_set_name is None:
        data_set_name = random.choice(all_data_sets)
    else:
        if data_set_name not in all_data_sets:
            raise Exception(f"Data set choice not valid: {data_set_name}")

    if data_set_name in seaborn_data_sets:
        if data_set_name == "mpg":
            data_set = sns.load_dataset(name="mpg").dropna().reset_index()
            predictors = [
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "origin",
                "name",
            ]
            response = "mpg"
        elif data_set_name == "tips":
            data_set = sns.load_dataset(name="tips").dropna().reset_index()
            predictors = [
                "total_bill",
                "sex",
                "smoker",
                "day",
                "time",
                "size",
            ]
            response = "tip"
        elif data_set_name == "titanic":
            data_set = sns.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "survived"
        elif data_set_name == "titanic_2":
            data_set = sns.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "alive"
    elif data_set_name in sklearn_data_sets:
        if data_set_name == "boston":
            data = datasets.load_boston()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)
            data_set["CHAS"] = data_set["CHAS"].astype(str)
        elif data_set_name == "diabetes":
            data = datasets.load_diabetes()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)
            data_set["gender"] = ["1" if i > 0 else "0" for i in data_set["sex"]]
        elif data_set_name == "breast_cancer":
            data = datasets.load_breast_cancer()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)

        data_set["target"] = data.target
        predictors = data.feature_names
        response = "target"

    print(f"Data set selected: {data_set_name}")
    return data_set, predictors, response


def check_list(list1, list2):
    """
    if list are none_type, will return empty string
    Used for calling all my methods even if list is empty
    """
    if list1 is None:
        list1 = [""]
    elif list2 is None:
        list2 = [""]
    return list1, list2


def make_clickable(val):
    # target _blank to open new window
    return '<a target="_blank" href="{}">{}</a>'.format(val, val)


def main():
    # getting current working dir
    plot_folder_dir = "/home/bioinfo/Desktop/html_plots_and_tables"

    # Getting DF, predictors, and response
    df, predictors, response = get_test_data_set()
    # read in object
    object = read_data(response=response, df=df, predictors=predictors)
    object.dataframe()

    # split predictor columns into respective groups
    # and response variable category
    continuous, categorical, boolean = object.checkColIsContOrCat()

    # Getting response type
    response_VarGroup = object.get_col_type(response)
    print("Response Type: ", response, response_VarGroup)

    # plotting continuous predictors with response
    # Also grabbing pvalues and tvalues from continuous responses
    stats_values, predictor_type, plot_paths, predictor_name, resp_name, resp_type = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for cont_pred in continuous:
        if continuous is None:
            continue
        if response_VarGroup == "continuous":
            test = Cat_vs_Cont(
                cont_pred, response=response, df=df, predictors=predictors
            ).contResponse_vs_contPredictor()
            _,_,_,file = test
            cont_name, t_value, p_value,_ = test
            stats_values.append((cont_name,t_value, p_value,))
            predictor_name.append(cont_pred)
            predictor_type.append("Continuous")
            plot_paths.append(file)
            resp_name.append(response)
            resp_type.append(response_VarGroup)

        elif response_VarGroup == "categorical":
            test = Cat_vs_Cont(
                cont_pred, response=response, df=df, predictors=predictors
            ).catResponse_vs_contPredictor()

            file = test; print("this is a file: ", file)
            predictor_name.append(cont_pred)
            predictor_type.append("Continuous")
            plot_paths.append(file)
            resp_name.append(response)
            resp_type.append(response_VarGroup)

        elif response_VarGroup == "boolean":
            test = Cat_vs_Cont(
                continuous=cont_pred,
                boolean=response,
                response=response,
                df=df,
                predictors=predictors,
            ).BoolResponse_vs_ContPredictor()
            cont_name, tval, pval, file = test
            stats_values.append((cont_name, tval, pval))
            predictor_name.append(cont_pred)
            predictor_type.append("Continuous")
            plot_paths.append(file)
            resp_name.append(response)
            resp_type.append(response_VarGroup)
        else:
            print(cont_pred, response_VarGroup)
            raise TypeError("invalid input...by me")
    
    # Plotting categorical predictors with response
    for cat_pred in categorical:
        if categorical is None:
            continue
        if response_VarGroup == "continuous":
            test = Cat_vs_Cont(
                categorical=cat_pred, response=response, df=df, predictors=predictors
            ).contResponse_vs_catPredictor()
            
            file = test; print("this is a file: ", file)
            predictor_name.append(cat_pred)
            predictor_type.append("Categorical")
            plot_paths.append(file)
            resp_name.append(response)
            resp_type.append(response_VarGroup)

        elif response_VarGroup in ("categorical", "boolean"):
            test = Cat_vs_Cont(
                categorical=cat_pred, response=response, df=df, predictors=predictors
            ).catResponse_vs_catPredictor()

            file = test; print("this is a file: ", file)
            predictor_name.append(cat_pred)
            predictor_type.append("Categorical")
            plot_paths.append(file)
            resp_name.append(response)
            resp_type.append(response_VarGroup)
        else:
            print(cat_pred, response_VarGroup)
            raise AttributeError(
                "Something is not being plotted correctly, issue with class?"
            )

    HW_4_html_df = pd.DataFrame(
        {
            "Predictor Type": predictor_type,
            "Predictor": predictor_name,
            "Response": resp_name,
            "Response type": resp_type,
            "Links to Plots": plot_paths,
        }
    )

    HW_4_html_df = HW_4_html_df.style.format({"Links to Plots": make_clickable})
    HW_4_html_df.to_html("../html_plots_and_tables/__HW4_plots.html", escape="html")

    # RF regressor, obtaining feature importance
    if response_VarGroup == "continuous":
        machineLearning = RF_importance(
            response=response, df=df, predictors=continuous, regressor=1
        )
        feature_ranking = machineLearning.RF_rank()
    elif response_VarGroup == "boolean":
        machineLearning = RF_importance(
            response=response, df=df, predictors=continuous, regressor=0
        )
        feature_ranking = machineLearning.RF_rank()
    else:
        print(response, response_VarGroup)
        raise AttributeError(
            "Only accounting for boolean and Continuous response variables"
        )

    # combining regression statistics with RF feature importance
    stats_df = pd.DataFrame(stats_values)
    stats_df.columns = ["Feature", "t-val", "p-val"]
    stats_df["Feature_Importance"] = np.array(
        feature_ranking["Feature_Importance"]
    ).tolist()

    stats_df.to_html("../html_plots_and_tables/__FeatureRanking.html")

    # ------ plotting the mean of response stuff ---------
    weighted_mr, unweighted_mr = [], []
    bin_table_links, plotly_plot_links = [], []
    pred_label, response_label = [], []
    for pred in predictors:
        mr_df = DiffMeanResponse(
            response=response, df=df, pred_input=pred
        ).Mean_Squared_DF()
        # print(pred, '\n', MeanResponseDF)
        mr_df.to_html(f"{plot_folder_dir}/{pred} MR_table.html")
        table_path = f"{plot_folder_dir}/{pred} MR_table.html"
        MeanPlots = DiffMeanResponse(
            response=response, df=df, pred_input=pred
        ).plot_Mean_diff()
        file = MeanPlots
        plot_path = f"{plot_folder_dir}/{file}"
        unweighted_mean = np.mean(mr_df["MeanSquaredDiff"])
        weighted_mean = np.mean(mr_df["MeanSquaredDiffWeighted"])
        weighted_mr.append(weighted_mean)
        unweighted_mr.append(unweighted_mean)
        bin_table_links.append(table_path)
        plotly_plot_links.append(plot_path)
        pred_label.append(pred)
        response_label.append(response)
    mr_report_df = pd.DataFrame(
        {
            "Response": response_label,
            "Predictor": pred_label,
            "WeightedMeanofResponse": weighted_mr,
            "UnWeightedMeanOfResponse": unweighted_mr,
            "LinksToBinReport": bin_table_links,
            "LinksToPlots": plotly_plot_links,
        }
    )
    mr_report_df = mr_report_df.style.format(
        {"LinksToBinReport": make_clickable, "LinksToPlots": make_clickable}
    )
    mr_report_df.to_html("../html_plots_and_tables/__MEANofRESPONSE_report.html", escape="html")

    # ----- getting Predictor correlation values -------

    # using reverse list of correlation
    reversed_cont = sorted(continuous, reverse=True)

    # checking lists if they are empty
    continuous, reversed_cont = check_list(continuous, reversed_cont)

    # getting corr stats for Continuous vs Continuous
    contVScont_stats = []
    for tupl in itertools.product(continuous, reversed_cont):
        corr_object = Correlation(
            response=response, df=df, a=tupl[0], b=tupl[1]
        ).cont_cont_Corr()
        contVScont_stats.append(corr_object)
    cont_corrDF = pd.DataFrame(
        contVScont_stats, columns=["Predictor 1", "Predictor 2", "Corr Coef", "Corr Type"]
    ).sort_values("Corr Coef", ascending=False)

    # Categorical vs Cont correlation statistics
    catVScont_stats = []
    continuous, categorical = check_list(continuous, categorical)
    # itertools product gives you every combination of
    # list elements
    for tupl in itertools.product(
        continuous, categorical
    ):  # tupl is a tuple hence tupl[0], tupl[1]
        corr_object = Correlation(
            response=response, df=df, a=tupl[0], b=tupl[1]
        ).cat_cont_correlation_ratio(
            df[tupl[1]].reset_index(drop=True), df[tupl[0]].reset_index(drop=True)
        )
        # needs to be in (b,a)
        catVScont_stats.append(corr_object)
    # reset_index is necessary to make sure index don't get messed up in calculations
    cat_contDF = pd.DataFrame(
        catVScont_stats, columns=["Predictor 1", "Predictor 2", "Corr Coef", "Corr Type"]
    ).sort_values("Corr Coef", ascending=False)

    # Categorical vs Categorical predictor correlation values.
    catVScat_stats = []
    reverse_cat = sorted(categorical, reverse=True)
    categorical, categorical = check_list(categorical, reverse_cat)
    for tupl in itertools.product(
        categorical, reverse_cat
    ):  # tupl is a tuple hence tupl[0], tupl[1]
        corr_object = Correlation(
            response=response, df=df, a=tupl[0], b=tupl[1]
        ).cat_correlation(
            df[tupl[1]].reset_index(drop=True), df[tupl[0]].reset_index(drop=True)
        )
        # needs to be in (b,a)
        catVScat_stats.append(corr_object)
    cat_corrDF = pd.DataFrame(
        catVScat_stats, columns=["Predictor 1", "Predictor 2", "Corr Coef", "Corr Type"]
    ).sort_values("Corr Coef", ascending=False)

    # making data frame with all the correlation rankings
    corr_report_df = pd.concat((cat_corrDF, cont_corrDF, cat_contDF), ignore_index=True
                    ).sort_values('Corr Coef', ascending=False)
    print(corr_report_df)
    corr_report_df.to_html("../html_plots_and_tables/__CorrelationRanking_report.html", escape="html")


    # ---- Plotting Correlation Matrix ---------
    ContContMatrix = Correlation(response=response, df=df).cont_vs_cont_matrix()
    cat_cont_matrixPlot = Correlation(df=df, response=response).cat_vs_cont_matrix(
        cat_contDF
    )
    cat_corrPlots = Correlation(df=df, response=response).cat_vs_cat_matrix(cat_corrDF)
    print("Using plot", cat_corrPlots, ContContMatrix, cat_cont_matrixPlot)
    # ----- Calculating Brute Force --------

    # BruteForce Cont/ Cat:
    cont_cat_brutDF = BruteForce(
        df=df, response=response, list1=categorical, list2=continuous
    ).BF_df()
    # print('BruteForce Cont/ Cat: \n',cont_cat_brutDF)

    # BruteForce Cat / Cat:
    cat_cat_brutDF = BruteForce(
        df=df, response=response, list1=categorical, list2=categorical
    ).BF_df()
    # print('Cat / Cat \n',cat_cat_brutDF)

    # BruteForce Cont / Cont:
    cont_cont_brutDF = BruteForce(
        df=df, response=response, list1=continuous, list2=continuous
    ).BF_df()
    # print("Cont / Cont \n",cont_cont_brutDF)

    # ------ Brute Force Matrix Plots ---------
    # BruteForce Matrix: Cat / Cont
    bf_cont_cat_plots = BruteForce(
        df=df, response=response, list1=continuous, list2=categorical
    ).plot_brutforce("Categorical vs Continuous")
    cont_cat_brutDF["BF Matrix Plot"] = bf_cont_cat_plots
    cont_cat_brutDF["BF Matrix Plot"] = f"{plot_folder_dir}/" + cont_cat_brutDF[
        "BF Matrix Plot"
    ].astype(str)
    cont_cat_brutDF = cont_cat_brutDF.style.format({"BF Matrix Plot": make_clickable})
    cont_cat_brutDF.to_html(f"{plot_folder_dir}/___BF_Cont_Cat_table.html", escape="html")

    # BruteForce Matrix: Cat / Cat
    bf_cat_cat_plots = BruteForce(
        df=df, response=response, list1=categorical, list2=categorical
    ).plot_brutforce("Categorical vs Categorical")

    cat_cat_brutDF["BF Matrix Plot"] = bf_cat_cat_plots
    cat_cat_brutDF["BF Matrix Plot"] = f"{plot_folder_dir}/" + cat_cat_brutDF[
        "BF Matrix Plot"
    ].astype(str)
    cat_cat_brutDF = cat_cat_brutDF.style.format({"BF Matrix Plot": make_clickable})
    cat_cat_brutDF.to_html(f"{plot_folder_dir}/___BF_Cat_Cat_table.html", escape="html")

    # BruteForce Matrix: Cont / Cont
    bf_cont_cont_plots = BruteForce(
        df=df, response=response, list1=continuous, list2=continuous
    ).plot_brutforce("Continuos vs Continuous")
    cont_cont_brutDF["BF Matrix Plot"] = bf_cont_cont_plots
    cont_cont_brutDF["BF Matrix Plot"] = f"{plot_folder_dir}/" + cont_cont_brutDF[
        "BF Matrix Plot"
    ].astype(str)
    cont_cont_brutDF = cont_cont_brutDF.style.format({"BF Matrix Plot": make_clickable})
    cont_cont_brutDF.to_html(f"{plot_folder_dir}/___BF_Cont_Cont_table.html", escape="html")

    print(
        """
        Hi professor, I want to mention a few things: \
        1. All of the reports are in a html file that start with '__blah'
        2. For some reasons the links to the plots work with Firefox web_browser and not chrome
        3. I wasn't sure if we were supposed to make links to the correlation matrix plots, so I just used fig_show()
            If you can....go easy on me.....have mercy...thanks
        """
    )

    return


if __name__ == "__main__":
    sys.exit(main())