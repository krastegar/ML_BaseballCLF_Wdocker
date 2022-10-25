import sys

import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.io import to_html
from scipy.stats import binned_statistic
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class read_data:
    def __init__(
        self,
        pathway="/home/bioinfo/Desktop/BDA_602/src/heart.csv",
        response="Sex", # --- code breaks when using sex as response
    ):
        self.pathway = pathway
        self.response = response
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
                    # print(self.columns[index], " Continuous")
                    continue
                elif data_type.dtype == "string" or data_type.dtype == "object":
                    cat_array.append(self.columns[index])
                    # print(self.columns[index], " Not Continuous or Bool")
                    continue
                else:  # Raise Error for unknown data types
                    raise TypeError("Unknown Data Type")
            else:
                bool_array.append(self.columns[index])
                # print(self.columns[index], " Boolean")
                continue
            # cat_array = cat_array + bool_array

        return cont_array, cat_array, bool_array
    
    def get_response_type(self):
        '''
        Creating a function to return the type of "response name" and "response type"
        '''
        cont_array, cat_array, bool_array = self.checkColIsContOrCat()
        if self.response in bool_array:
            group_resp = "boolean"
        elif self.response in cat_array:
            group_resp = "categorical"
        elif self.response in cont_array:
            group_resp = "continuous"
        else:
            raise TypeError("Unknown Input")

        return self.response, group_resp


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
        #df[self.boolean] = df[self.boolean].astype(int) # changing booleans into ints

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
    def __init__(self, predictor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictors = predictor

    def bin_params(self, bins_edge):
        bin_lower, bin_upper, bin_center = [], [], []
        # have to start at index 1 and then subtract 1 to get index 0
        for edge in range(1, len(bins_edge)): 
            lower = bins_edge[edge-1]
            upper = bins_edge[edge]
            center = (upper+lower)/2 
            bin_lower.append(lower)
            bin_upper.append(upper)
            bin_center.append(center)
        return bin_lower, bin_upper, bin_center

    def squared_diffs(self, 
                        bin_means, 
                        bins_edge, 
                        pop_mean_response,
                        population_proportion
                        ):

        mean_square_diff, weighted_square_diff = [], []
        for mean in range(len(bin_means)):
            UnWeighted=(1/len(bins_edge))*(bin_means[mean]-pop_mean_response)**2 
            weighted = (population_proportion[mean])*(bin_means[mean]-pop_mean_response)**2
            mean_square_diff.append(UnWeighted)
            weighted_square_diff.append(weighted)

        return mean_square_diff, weighted_square_diff

    def Mean_Squared_DF(self):

        
        # Reading in df
        df = self.ChangeBinaryToBool()
       
        #--------------FOUND A WAY TO DO MEAN OF RESPONSE ------------------------------
        if df[self.predictors].dtype != ('int64', 'float64'):
            
            # determining if response is cat or cont
            # if cat use label encoder to make bins 
            unique_len = len(np.unique(df[self.predictors]))            
            le = LabelEncoder()
            label_fit=le.fit_transform(df[self.predictors])
            mean_stat = binned_statistic(label_fit, df[self.response],
                                statistic='mean', 
                                bins=unique_len
                                )
            bin_count, bins_edge = np.histogram(label_fit, bins=unique_len)
            population_proportion = bin_count/len(df)
            bin_means = mean_stat.statistic
            pop_mean_response = np.mean(df[self.response])
            
            bin_lower, bin_upper, bin_center = self.bin_params(bins_edge)
            mean_square_diff, weighted_diff = self.squared_diffs(bins_edge=bins_edge,
                                                                bin_means=bin_means,
                                                                pop_mean_response=pop_mean_response,
                                                                population_proportion=population_proportion)
            bin_label = np.unique(le.inverse_transform(label_fit))

        else: 
            mean_stat = binned_statistic(df[self.predictors], df[self.response],
                                statistic='mean', 
                                bins=9
                                )
            pop_mean_response = np.mean(df[self.response])
            bin_count, bins_edge = np.histogram(df[self.predictors], bins=9)
            population_proportion = bin_count/len(df)
            bin_means = mean_stat.statistic
            bin_lower, bin_upper, bin_center = self.bin_params(bins_edge)
            mean_square_diff, weighted_diff = self.squared_diffs(bins_edge=bins_edge,
                                                                bin_means=bin_means,
                                                                pop_mean_response=pop_mean_response,
                                                                population_proportion=population_proportion)
            bin_label = np.unique(mean_stat.binnumber)
        
        mean_diff_df = pd.DataFrame(
                {'Bin': bin_label,
                'LowerBin': bin_lower,
                "UpperBin": bin_upper,
                "BinCenters": bin_center,
                "BinCount": bin_count,
                "BinMeans": bin_means, 
                "PopulationMean": pop_mean_response,
                "PopulationProportion": population_proportion,
                "MeanSquaredDiff": mean_square_diff,
                "MeanSquaredDiffWeighted": weighted_diff}
                    )
        mean_diff_df = mean_diff_df .sort_values("MeanSquaredDiffWeighted", ascending=False).reset_index(drop=True)
        return mean_diff_df
        
    def plot_Mean_diff(self):
        
        mean_diff_df = self.Mean_Squared_DF()
        fig = go.Figure(
            layout=go.Layout(
            title="NOT SURE",
            yaxis2=dict(overlaying="y", side="right"),
                )
            )
        fig.add_trace(
            go.Bar(name="counts", x=mean_diff_df["BinCenters"], y=mean_diff_df["BinCount"], yaxis="y1"),
            )
        fig.add_trace(
            go.Scatter(
                name="Pop-Mean", x=mean_diff_df["BinCenters"], y=mean_diff_df["PopulationMean"], yaxis="y2"
            )
        )      
        fig.add_trace(
            go.Scatter(
                name="Squared diff",
                x=mean_diff_df["BinCenters"],
                y = mean_diff_df['BinMeans'],
                yaxis="y2",
            )
        )
        fig.update_layout(
            title=f"Difference w/ Mean Response: {self.predictors}"
        )
        fig.update_layout(
            xaxis_title="Predictor", yaxis_title="Population", yaxis2_title="Response"

        )
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(tickvals=mean_diff_df['BinCenters'],
                        ticktext=mean_diff_df['Bin'])
        fig.show()
        return


def main():

    # read in object and dataframe
    object = read_data()

    # -- testing DF w/ Mean Response
    bins= DiffMeanResponse('FastingBS')
    msd = bins.Mean_Squared_DF()
    print(msd)
    plots = bins.plot_Mean_diff()
    print(object.get_response_type())
    # split predictor columns into respective groups
    # and response variable category

    continuous, categorical, boolean = object.checkColIsContOrCat()
    response = object.get_response_type()
    
    '''
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
    feature_ranking = machineLearning.RF_regressor()
    
    # combining regression statistics with RF feature importance
    stats_df = pd.DataFrame(stats_values)
    stats_df.columns=['Feature', 't-val', 'p-val']
    stats_df['Feature_Importance'] = np.array(feature_ranking['Feature_Importance']).tolist()
    print(stats_df)
    '''
    return


if __name__ == "__main__":
    sys.exit(main())
