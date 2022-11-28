from midterm import BruteForce, make_clickable, read_data


class BF_summary(read_data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = self.ChangeBinaryToBool()

    def bruteforce_df(self):

        continuous, categorical, _ = self.checkColIsContOrCat()
        # BruteForce Cont/ Cat:
        cont_cat_brutDF = BruteForce(
            df=self.df, response=self.response, list1=categorical, list2=continuous
        ).BF_df()

        # BruteForce Cat / Cat:
        cat_cat_brutDF = BruteForce(
            df=self.df, response=self.response, list1=categorical, list2=categorical
        ).BF_df()

        # BruteForce Cont / Cont:
        cont_cont_brutDF = BruteForce(
            df=self.df, response=self.response, list1=continuous, list2=continuous
        ).BF_df()

        return cont_cont_brutDF, cont_cat_brutDF, cat_cat_brutDF

    def bruteForce_summary(self):

        # importing lists and dataframes to input into bruteforce class from midterm
        cont_cont_brutDF, cont_cat_brutDF, cat_cat_brutDF = self.bruteforce_df()
        continuous, categorical, _ = self.checkColIsContOrCat()

        plot_folder_dir = "/home/bioinfo/Desktop/html_plots_and_tables"
        bf_cont_cat_plots = BruteForce(
            df=self.df, response=self.response, list1=continuous, list2=categorical
        ).plot_brutforce("Categorical vs Continuous")
        cont_cat_brutDF["BF Matrix Plot"] = bf_cont_cat_plots
        cont_cat_brutDF["BF Matrix Plot"] = f"{plot_folder_dir}/" + cont_cat_brutDF[
            "BF Matrix Plot"
        ].astype(str)
        cont_cat_brutDF = cont_cat_brutDF.style.format(
            {"BF Matrix Plot": make_clickable}
        )
        cont_cat_brutDF.to_html(
            f"{plot_folder_dir}/___BF_Cont_Cat_table.html", escape="html"
        )

        # BruteForce Matrix: Cat / Cat
        bf_cat_cat_plots = BruteForce(
            df=self.df, response=self.response, list1=categorical, list2=categorical
        ).plot_brutforce("Categorical vs Categorical")

        cat_cat_brutDF["BF Matrix Plot"] = bf_cat_cat_plots
        cat_cat_brutDF["BF Matrix Plot"] = f"{plot_folder_dir}/" + cat_cat_brutDF[
            "BF Matrix Plot"
        ].astype(str)
        cat_cat_brutDF = cat_cat_brutDF.style.format({"BF Matrix Plot": make_clickable})
        cat_cat_brutDF.to_html(
            f"{plot_folder_dir}/___BF_Cat_Cat_table.html", escape="html"
        )

        # BruteForce Matrix: Cont / Cont
        bf_cont_cont_plots = BruteForce(
            df=self.df, response=self.response, list1=continuous, list2=continuous
        ).plot_brutforce("Continuos vs Continuous")
        cont_cont_brutDF["BF Matrix Plot"] = bf_cont_cont_plots
        cont_cont_brutDF["BF Matrix Plot"] = f"{plot_folder_dir}/" + cont_cont_brutDF[
            "BF Matrix Plot"
        ].astype(str)
        cont_cont_brutDF = cont_cont_brutDF.style.format(
            {"BF Matrix Plot": make_clickable}
        )
        cont_cont_brutDF.to_html(
            f"{plot_folder_dir}/___BF_Cont_Cont_table.html", escape="html"
        )

        return
