import sys

import pandas as pd
import numpy as np
from midterm import (
    read_data, 
    RF_importance
)
from Cat_vs_Cont_stats_and_plots import Cont_Cat_stats_plots

class RF_ranking_stat(RF_importance, read_data): 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def ranking(self):
        
        # Get response type, and list of columns in each category
        response_VarGroup = self.get_col_type(self.response)
        continuous, _, _ = self.checkColIsContOrCat()
        # RF regressor, obtaining feature importance
        
        if response_VarGroup == "continuous":
            machineLearning = RF_importance(
                response=self.response, df=self.df, predictors=continuous, regressor=1
            )
            feature_ranking = machineLearning.RF_rank()
        elif response_VarGroup == "boolean":
            machineLearning = RF_importance(
                response=self.response, df=self.df, predictors=continuous, regressor=0
            )
            feature_ranking = machineLearning.RF_rank()
        else:
            print(self.response, response_VarGroup)
            raise AttributeError(
                "Only accounting for boolean and Continuous response variables"
            )

        # combining regression statistics with RF feature importance
        stats_values,_,_,_,_,_= Cont_Cat_stats_plots(response=self.response, 
                                df=self.df, 
                                predictors=self.predictors).predictor_plots()
        stats_df = pd.DataFrame(stats_values)
        stats_df.columns = ["Feature", "t-val", "p-val"]
        stats_df["Feature_Importance"] = np.array(
            feature_ranking["Feature_Importance"]
        ).tolist()

        stats_df.to_html("../html_plots_and_tables/__FeatureRanking.html")
        return 