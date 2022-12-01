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
        plot_folder_dir = f"{curr_path}/html_plots_and_tables"
        weighted_mr, unweighted_mr = [], []
        bin_table_links, plotly_plot_links = [], []
        pred_label, response_label = [], []

        for pred in self.predictors:
            mr_df = DiffMeanResponse(
                response=self.response, df=self.df, pred_input=pred
            ).Mean_Squared_DF()
            # print(pred, '\n', MeanResponseDF)
            mr_df.to_html(f"{plot_folder_dir}/{pred} MR_table.html")
            table_path = f"{plot_folder_dir}/{pred} MR_table.html"
            MeanPlots = DiffMeanResponse(
                response=self.response, df=self.df, pred_input=pred
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
        mr_report_df = mr_report_df.style.format(
            {"LinksToBinReport": make_clickable, "LinksToPlots": make_clickable}
        )
        mr_report_df.to_html(
            "html_plots_and_tables/__MEANofRESPONSE_report.html", escape="html"
        )
        return
