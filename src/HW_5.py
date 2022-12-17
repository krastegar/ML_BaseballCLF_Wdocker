import sys

from BrutF import BF_summary
from Cat_vs_Cont_stats_and_plots import Cont_Cat_stats_plots
from Correlation_Metrics import Corr_report
from dataset_loader import TestDatasets
from Mean_of_Response import DifferenceOfMeans
from midterm import read_data
from RandomForestRanking import RF_ranking_stat
from split import TrainTestModel
from sql_dataLoader import sqlPandasSpark


def main():

    # getting dataframe of my features from sql
    df = sqlPandasSpark().sql_to_pandas()
    response = "Home_Team_Wins"
    predictors = [i for i in list(df.columns) if response not in i]
    dataset = TestDatasets()
    (
        _,
        _,
        _,
    ) = dataset.get_test_data_set()  # replace '_' w/ df, predictors, response
    # place holder for testing diff df
    # print(predictors)
    # HW_4 plots
    hw_4 = Cont_Cat_stats_plots(response=response, df=df, predictors=predictors)
    _, _, _, _, _, _ = hw_4.predictor_plots()
    _ = hw_4.plot_summary_html()

    # RandomForest Ranking Report
    _ = RF_ranking_stat(response=response, df=df, predictors=predictors).ranking()

    # Mean of Response Report: Plots and Calculations
    _ = DifferenceOfMeans(response=response, df=df, predictors=predictors).plot_MoR()

    # Correlation Report: Plots (fig.show()) and Calculations
    _ = Corr_report(
        response=response, df=df, predictors=predictors
    ).corr_summary_report()

    _ = Corr_report(response=response, df=df, predictors=predictors).corr_matrix_plots()

    # BruteForce Report: Plots and Calculations
    _ = BF_summary(response=response, df=df, predictors=predictors).bruteForce_summary()

    # Train test split

    continuous, categorical, _ = read_data(
        response=response, df=df, predictors=predictors
    ).checkColIsContOrCat()
    model_obj = TrainTestModel(
        response=response,
        df=df,
        predictors=predictors,
        list1=continuous,
        list2=categorical,
        local_date=0,  # change this for non sql data
    )
    _ = model_obj.trainer_tester()

    _ = model_obj.Feature_Importance()

    _ = model_obj.model_evaluation()

    _ = model_obj.plot_feature_rank()
    return


if __name__ == "__main__":
    sys.exit(main())
