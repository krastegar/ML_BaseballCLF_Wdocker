from argparse import ArgumentError
from nis import cat
import random
import sys
from typing import List  # might not need this

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.io import to_html
from scipy.stats import binned_statistic
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class read_data:
    def __init__(
        self,
        response,
        df,
        predictors=None
    ):
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
                elif unique[0] in ('n','N','no','No') and unique[1] in ('y','Y', 'yes', 'Yes'):
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
        cont_array, cat_array, bool_array = [], [], []

        # Make sure that binary response column is changed into a boolean
        df = self.ChangeBinaryToBool()  # -- method introduces bug in my code

        # Looping through each column and determing conditions
        # to put columns into respective groups
        list_of_columns = list(df.columns)
        for index, column in enumerate(list_of_columns): #df.columns is a list of columns
            data_type = df[column]
            col_name = self.columns[index]
            if data_type.dtype != "bool":
                if data_type.dtype in ["float64","int64"]:
                    cont_array.append(col_name)
                    # print(col_name, " Continuous")
                    continue
                elif data_type.dtype in ["string", "object", "category"]:
                    cat_array.append(col_name)
                    # print(col_name, " Not Continuous or Bool")
                    continue
                else:  # Raise Error for unknown data types
                    raise TypeError("Error in column classification: Unknown column dtype")
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

    def __init__(self, continuous=None, categorical=None, boolean=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.categorical = categorical
        self.continuous = continuous
        self.boolean = boolean

    def catResponse_vs_contPredictor(self):
        fig_2 = go.Figure()
        fig_2.add_trace(
            go.Violin(
                x=self.df[self.response],
                y=self.df[self.continuous],
                name=f"{self.df[self.response].name} vs {self.df[self.continuous].name}",
                box_visible=True,
                meanline_visible=True,
            )
        )
        fig_2.update_layout(
            title=f"{self.df[self.response].name} vs {self.df[self.continuous].name}",
            xaxis_title=f"{self.df[self.response].name}",
            yaxis_title=f"{self.df[self.continuous].name}",
        )
        fig_2.show()
        fig_2.write_html(
            file=f"ViolinPlots: {self.df[self.continuous].name}",
            include_plotlyjs="cdn",
        )
        return

    def contResponse_vs_catPredictor(self):

        hist_data = [self.df[self.response]]
        group_labels = [f"{self.df[self.categorical].name}"]  # name of the dataset
        fig = ff.create_distplot(hist_data, group_labels)
        fig.show()

        fig.write_html(
            file=f"ViolinPlots pt 2: {self.df[self.response].name}",
            include_plotlyjs="cdn",
        )
        return

    def catResponse_vs_catPredictor(self):

        # I am using self.continuous as a place holder for another categorical variaible
        conf_matrix = confusion_matrix(
            self.df[self.response], self.df[self.categorical]
        )
        fig_no_relationship = go.Figure(
            data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
        )
        fig_no_relationship.update_layout(
            title="Categorical Predictor by Categorical Response",
            xaxis_title=f"{self.df[self.response].name}",
            yaxis_title=f"{self.df[self.continuous].name}",
        )
        fig_no_relationship.show()

        fig_no_relationship.write_html(
            file=f"HeatMap: {self.df[self.continuous].name} vs {self.df[self.response].name} ",
            include_plotlyjs="cdn",
        )
        return

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
            title=f"Variable: {self.df[self.continuous].name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"{self.df[self.continuous].name}",
            yaxis_title=f"{self.df[self.response].name}",
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
        fig = px.scatter(x=self.df[self.continuous], y=df[self.response].astype(int), trendline="ols")
        fig.update_layout(
            title=f"Variable: {df[self.continuous].name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"{df[self.continuous].name}",
            yaxis_title=f"{df[self.response].name}",
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

    def __init__(self, regressor = 1,continuous=None, categorical=None, boolean=None, *args, **kwargs):
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
        if self.regressor ==  1:
            RandForest_regressor = RandomForestRegressor(max_depth=4, random_state=42)
            RF_clf = RandForest_regressor.fit(X_train, y_train)
            df_feature_importance = pd.DataFrame(
                RF_clf.feature_importances_, index=X.columns, columns=["Feature_Importance"]
            ).sort_values("Feature_Importance", ascending=False)
        elif self.regressor == 0:
            RandForest_regressor = RandomForestRegressor(max_depth=4, random_state=42)
            RF_clf = RandForest_regressor.fit(X_train, y_train)
            df_feature_importance = pd.DataFrame(
                RF_clf.feature_importances_, index=X.columns, columns=["Feature_Importance"]
            ).sort_values("Feature_Importance", ascending=False)
        
        else: raise ArgumentError("Invalid option inputted")


        return df_feature_importance


class DiffMeanResponse(read_data):
    def __init__(self, predictor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictors = predictor

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
        '''
        Mean of Response calculations and plots. This class is to be used in 
        a for loop with individual response and columns 
        '''
        # Reading in df
        df = self.ChangeBinaryToBool()

        # Getting response type
        response_type = self.get_col_type(self.response)
        predictor_type = self.get_col_type(self.predictors)
        print(predictor_type)

        # Predictor and response Data
        predictor_data, response_data = df[self.predictors], df[self.response]

        # --------------FOUND A WAY TO DO MEAN OF RESPONSE ------------------------------
        # This first condition only deals with categorical predictors not categorical response
        if response_type == "boolean":

            # if cat use label encoder to make bins
            unique_len = len(np.unique(predictor_data))
            le = LabelEncoder()
            label_fit = le.fit_transform(predictor_data)
            mean_stat = binned_statistic(
                label_fit, response_data, statistic="mean", bins=unique_len
            )
            bin_count, bins_edge = np.histogram(label_fit, bins=unique_len)
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
            bin_label = np.unique(le.inverse_transform(label_fit))

        elif (
            response_type == "continuous"
        ):  # this breaks when response type is continuous and predictor is boolean?
            if predictor_type == "categorical":
                le = LabelEncoder()
                label_fit = le.fit_transform(predictor_data)
                predictor_data = label_fit
                bin_size = len(np.unique(predictor_data))
            elif predictor_type == "boolean":
                bin_size = len(np.unique(predictor_data))
                predictor_data = predictor_data.astype(int)
            elif predictor_type == "continuous":
                bin_size = 9
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
            bin_label = np.unique(df[self.predictors])

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
        fig.update_layout(title=f"Difference w/ Mean Response: {self.predictors}")
        fig.update_layout(
            xaxis_title="Predictor", yaxis_title="Population", yaxis2_title="Response"
        )
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(
            tickvals=mean_diff_df["BinCenters"], ticktext=mean_diff_df["Bin"]
        )
        fig.show()
        return


def get_test_data_set(data_set_name: str = None)-> (pd.DataFrame, List[str], str):
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


def main():

    # Getting DF, predictors, and response 
    df, predictors, response = get_test_data_set()
    
    # read in object
    object = read_data(response= response, 
                        df=df, 
                        predictors=predictors)
    object.dataframe()

    # split predictor columns into respective groups
    # and response variable category
    continuous, categorical, boolean = object.checkColIsContOrCat()
    print("continuous: \n", continuous,
        "categorical: \n", categorical,
        "boolean: \n", boolean)
    # Getting response type
    response_VarGroup = object.get_col_type(response)

    # plotting continuous predictors with response
    # Also grabbing pvalues and tvalues from continuous responses 
    stats_values = []
    for cont_pred in continuous:
        if continuous is None:
            continue
        if response_VarGroup == "continuous":
            test = Cat_vs_Cont(cont_pred, 
            response= response, 
            df=df, 
            predictors=predictors).contResponse_vs_contPredictor()
            stats_values.append(test)
        
        elif response_VarGroup == "categorical":
            test = Cat_vs_Cont(
            cont_pred,
            response= response, 
            df=df, 
            predictors=predictors).contResponse_vs_catPredictor()
        
        elif response_VarGroup == "boolean":
            test = Cat_vs_Cont(
                continuous=cont_pred, 
                boolean=response, 
                response= response, 
                df=df, 
                predictors=predictors).BoolResponse_vs_ContPredictor()
            stats_values.append(test)
        else:
            print(cont_pred, response_VarGroup)
            raise TypeError("invalid input...by me")
    
    # Plotting categorical predictors with response
    for cat_pred in categorical:
        if categorical is None: 
            continue
        if response_VarGroup == "continuous":
            test = Cat_vs_Cont(categorical=cat_pred, 
            response= response, 
            df=df, 
            predictors=predictors).contResponse_vs_catPredictor()
        
        elif response_VarGroup in ("categorical", "boolean"):
            test = Cat_vs_Cont(categorical=cat_pred,
            response= response, 
            df=df, 
            predictors=predictors).catResponse_vs_catPredictor()
        
        else:
            print(cat_pred, response_VarGroup)
            raise AttributeError("Something is not being plotted correctly, issue with class?")

    # RF regressor, obtaining feature importance
    if response_VarGroup == 'continuous':
        machineLearning = RF_importance(response= response, 
                                        df=df, 
                                        predictors=continuous,
                                        regressor=1)
        feature_ranking = machineLearning.RF_rank()
    elif response_VarGroup == 'boolean':
        machineLearning = RF_importance(response= response, 
                            df=df, 
                            predictors=continuous,
                            regressor=0)
        feature_ranking = machineLearning.RF_rank()
    else: pass
    
    # combining regression statistics with RF feature importance
    stats_df = pd.DataFrame(stats_values)
    stats_df.columns=['Feature', 't-val', 'p-val']
    stats_df['Feature_Importance'] = np.array(feature_ranking['Feature_Importance']).tolist()
    print(stats_df)
    
    return


if __name__ == "__main__":
    sys.exit(main())
