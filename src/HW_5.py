import sys

import pandas as pd
import numpy as np
from sql_dataLoader import sqlPandasSpark
from midterm import read_data
from Cat_vs_Cont_stats_and_plots import Cont_Cat_stats_plots
from RandomForestRanking import RF_ranking_stat
from Mean_of_Response import DifferenceOfMeans
from Correlation_Metrics import Corr_report
from BrutF import BF_summary
from dataset_loader import TestDatasets
from split import TrainTestModel


def main():
    
    # getting dataframe of my features from sql
    df = sqlPandasSpark().sql_to_pandas()
    response = 'HomeWins'
    predictors = [i for i in list(df.columns) if not response in i]
    dataset = TestDatasets()
    _, _, _= dataset.get_test_data_set() # df, predictors, response
                                         # place holder for testing diff df
    # HW_4 plots
    hw_4 = Cont_Cat_stats_plots(response=response, 
                                df=df, 
                                predictors=predictors
                                )
    hw_4_plots = hw_4.predictor_plots()
    hw_4_report = hw_4.plot_summary_html()

    # RandomForest Ranking Report 
    RF_report = RF_ranking_stat(response=response, 
                                df=df, 
                                predictors=predictors
                                ).ranking()

    # Mean of Response Report: Plots and Calculations 
    mean_of_response = DifferenceOfMeans(response=response, 
                                df=df, 
                                predictors=predictors
                                ).plot_MoR()

    # Correlation Report: Plots (fig.show()) and Calculations 
    correlations_report = Corr_report(response=response, 
                                df=df, 
                                predictors=predictors
                                ).corr_summary_report()
    
    correlations_plot = Corr_report(response=response, 
                                df=df, 
                                predictors=predictors
                                ).corr_matrix_plots()
    
    # BruteForce Report: Plots and Calculations
    Brute_Report = BF_summary(response=response, 
                                df=df, 
                                predictors=predictors
                                ).bruteForce_summary()

    # Train test split 
    _ = 'dummy'
    continuous, categorical, _ = read_data(response=response, 
                                df=df, 
                                predictors=predictors
                                ).checkColIsContOrCat()
    model_test = TrainTestModel(response=response, 
                                df=df, 
                                predictors=predictors,
                                list1 = continuous,
                                list2 = categorical,
                                local_date=1
                                ).trainer_tester()
    # Analysis:
    print("""
    Comparing precision, recall, f1-score, and accuracy. 
    We can see that the gradient boosted model has scored higher on
    all metrics. 
    """)

    return


if __name__ == "__main__":
    sys.exit(main())
