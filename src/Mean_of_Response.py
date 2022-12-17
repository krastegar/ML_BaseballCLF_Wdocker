import numpy as np
import pandas as pd

from midterm import DiffMeanResponse, make_clickable, read_data


class DifferenceOfMeans(read_data):
    def __init__(self, pred_input=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred = pred_input
        self.df = self.ChangeBinaryToBool()

    # ------ plotting the mean of response stuff ---------
    def plot_MoR(self):
        curr_path = self.get_workingDir()
        plot_folder_dir = f"{curr_path}html_plots_and_tables/subplots"
        weighted_mr, unweighted_mr = [], []
        bin_table_links, plotly_plot_links = [], []
        pred_label, response_label = [], []

        for pred in self.predictors:

            # This is to get the mean of response ranking tables, i.e) pandas df
            # for each predictor (pred)
            mr_df = DiffMeanResponse(
                response=self.response, df=self.df, pred_input=pred
            ).Mean_Squared_DF()

            # creating html table from mean of response ranking tables
            # for each predictor
            mr_df.to_html(f"{plot_folder_dir}/{pred} MR_table.html")

            # getting relative paths for each table to put into final report
            # table
            table_path = f"{pred} MR_table.html"

            # Getting file paths to plots that were generated in mean of response
            # analysis
            MeanPlots = DiffMeanResponse(
                response=self.response, df=self.df, pred_input=pred
            ).plot_Mean_diff()
            file = MeanPlots
            # calculating summations of squared differences
            unweighted_mean = np.sum(mr_df["MeanSquaredDiff"])
            weighted_mean = np.sum(mr_df["MeanSquaredDiffWeighted"])

            # Puttting all file paths to plots and tables into one df with
            # the rest of the calculations
            weighted_mr.append(weighted_mean)
            unweighted_mr.append(unweighted_mean)
            bin_table_links.append(table_path)
            plotly_plot_links.append(file)
            pred_label.append(pred)
            response_label.append(self.response)
        mr_report_df = pd.DataFrame(
            {
                "Response": response_label,
                "Predictor": pred_label,
                "WeightedMeanofResponse": weighted_mr,
                "UnWeightedMeanOfResponse": unweighted_mr,
                "LinksToBinReport": bin_table_links,
                "LinksToPlots": plotly_plot_links,
            }
        ).sort_values("WeightedMeanofResponse", ascending=False)

        # making the links clickable
        mr_report_df = mr_report_df.style.format(
            {"LinksToBinReport": make_clickable, "LinksToPlots": make_clickable}
        )
        # making final df report into an html table
        mr_report_df.to_html(
            f"{curr_path}html_plots_and_tables/__MEANofRESPONSE_report.html",
            escape="html",
        )
        return
