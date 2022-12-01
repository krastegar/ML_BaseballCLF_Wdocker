import itertools

import pandas as pd

from midterm import Correlation, check_list, read_data


class Corr_report(read_data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = self.ChangeBinaryToBool()

    def corr_calculation(self):

        continuous, categorical, boolean = self.checkColIsContOrCat()
        # using reverse list of correlation
        reversed_cont = sorted(continuous, reverse=True)

        # checking lists if they are empty
        continuous, reversed_cont = check_list(continuous, reversed_cont)

        # getting corr stats for Continuous vs Continuous
        contVScont_stats = []
        for tupl in itertools.product(continuous, reversed_cont):
            corr_object = Correlation(
                response=self.response, df=self.df, a=tupl[0], b=tupl[1]
            ).cont_cont_Corr()
            contVScont_stats.append(corr_object)
        cont_corrDF = pd.DataFrame(
            contVScont_stats,
            columns=["Predictor 1", "Predictor 2", "Corr Coef", "Corr Type"],
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
                response=self.response, df=self.df, a=tupl[0], b=tupl[1]
            ).cat_cont_correlation_ratio(
                self.df[tupl[1]].reset_index(drop=True),
                self.df[tupl[0]].reset_index(drop=True),
            )
            # needs to be in (b,a)
            catVScont_stats.append(corr_object)
        # reset_index is necessary to make sure index don't get messed up in calculations
        cat_contDF = pd.DataFrame(
            catVScont_stats,
            columns=["Predictor 1", "Predictor 2", "Corr Coef", "Corr Type"],
        ).sort_values("Corr Coef", ascending=False)

        # Categorical vs Categorical predictor correlation values.
        catVScat_stats = []
        reverse_cat = sorted(categorical, reverse=True)
        categorical, categorical = check_list(categorical, reverse_cat)
        for tupl in itertools.product(
            categorical, reverse_cat
        ):  # tupl is a tuple hence tupl[0], tupl[1]
            corr_object = Correlation(
                response=self.response, df=self.df, a=tupl[0], b=tupl[1]
            ).cat_correlation(
                self.df[tupl[1]].reset_index(drop=True),
                self.df[tupl[0]].reset_index(drop=True),
            )
            # needs to be in (b,a)
            catVScat_stats.append(corr_object)
        cat_corrDF = pd.DataFrame(
            catVScat_stats,
            columns=["Predictor 1", "Predictor 2", "Corr Coef", "Corr Type"],
        ).sort_values("Corr Coef", ascending=False)
        return cat_contDF, cat_corrDF, cont_corrDF

    def corr_summary_report(self):
        cat_contDF, cat_corrDF, cont_corrDF = self.corr_calculation()
        # making data frame with all the correlation rankings
        corr_report_df = pd.concat(
            (cat_corrDF, cont_corrDF, cat_contDF), ignore_index=True
        ).sort_values("Corr Coef", ascending=False)
        print(corr_report_df)
        curr_path = self.get_workingDir()
        corr_report_df.to_html(
            f"{curr_path}/html_plots_and_tables/__CorrelationRanking_report.html",
            escape="html",
        )
        return

    def corr_matrix_plots(self):
        cat_contDF, cat_corrDF, cont_corrDF = self.corr_calculation()
        _ = Correlation(response=self.response, df=self.df).cont_vs_cont_matrix()
        _ = Correlation(df=self.df, response=self.response).cat_vs_cont_matrix(
            cat_contDF
        )
        _ = Correlation(df=self.df, response=self.response).cat_vs_cat_matrix(
            cat_corrDF
        )
        return
