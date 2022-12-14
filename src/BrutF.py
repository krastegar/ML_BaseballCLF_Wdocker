from midterm import BruteForce, make_clickable, read_data


class BF_summary(read_data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = self.ChangeBinaryToBool()

    def bruteforce_df(self):

        """
        The purpose of this function is to get BruteForce Df from midterm
        BruteForce class
        """

        """
        THERE IS A HUGE ISSUE WITH THE PLOT LINKS
        THEY ARE NOT IN THE SAME ORDER AS RANKINGS

        LOOK AT MEAN OF RESPONSE TO FIX THIS ISSUE
        """
        continuous, categorical, _ = self.checkColIsContOrCat()

        # BruteForce Cont/ Cat:
        cont_cat_brutDF = BruteForce(
            df=self.df, response=self.response, list1=categorical, list2=continuous
        ).BF_df(title="Continuous vs Categorical")

        # BruteForce Cat / Cat:
        cat_cat_brutDF = BruteForce(
            df=self.df, response=self.response, list1=categorical, list2=categorical
        ).BF_df(title="Categorical vs Categorical")

        # BruteForce Cont / Cont:
        cont_cont_brutDF = BruteForce(
            df=self.df, response=self.response, list1=continuous, list2=continuous
        ).BF_df(title="Continuous vs Continuous")

        return cont_cont_brutDF, cont_cat_brutDF, cat_cat_brutDF

    def bruteForce_summary(self):

        # getting brut force dataframes from another method in this same class
        # bruteforce_df
        cont_cont_brutDF, cont_cat_brutDF, cat_cat_brutDF = self.bruteforce_df()

        # getting full destination path for result plots
        curr_path = self.get_workingDir()
        plot_folder_dir = f"{curr_path}/html_plots_and_tables"

        # ------BruteForce Matrix: Cont / Cat ---------
        # Adding full path to plots
        cont_cat_brutDF["LinksToPlots"] = f"{plot_folder_dir}/" + cont_cat_brutDF[
            "LinksToPlots"
        ].astype(str)

        # Making the links clickable
        cont_cat_brutDF.sort_values("Weighted", ascending=False)
        cont_cat_brutDF = cont_cat_brutDF.style.format({"LinksToPlots": make_clickable})
        cont_cat_brutDF.to_html(
            f"{plot_folder_dir}/___BF_Cont_Cat_table.html", escape="html"
        )

        # ------BruteForce Matrix: Cat / Cat ----

        cat_cat_brutDF["LinksToPlots"] = f"{plot_folder_dir}/" + cat_cat_brutDF[
            "LinksToPlots"
        ].astype(str)
        cat_cat_brutDF = cat_cat_brutDF.style.format({"LinksToPlots": make_clickable})
        cat_cat_brutDF.to_html(
            f"{plot_folder_dir}/___BF_Cat_Cat_table.html", escape="html"
        )

        # BruteForce Matrix: Cont / Cont (repeating process above for all categories)
        cont_cont_brutDF["LinksToPlots"] = f"{plot_folder_dir}/" + cont_cont_brutDF[
            "LinksToPlots"
        ].astype(str)
        cont_cont_brutDF = cont_cont_brutDF.style.format(
            {"LinksToPlots": make_clickable}
        )
        cont_cont_brutDF.to_html(
            f"{plot_folder_dir}/___BF_Cont_Cont_table.html", escape="html"
        )

        return
