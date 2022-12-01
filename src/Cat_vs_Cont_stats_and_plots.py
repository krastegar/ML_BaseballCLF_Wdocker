import pandas as pd

from midterm import Cat_vs_Cont, make_clickable, read_data


class Cont_Cat_stats_plots(
    Cat_vs_Cont, read_data
):  # I think order matters for double inheritance
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = self.ChangeBinaryToBool()

    def predictor_plots(self):
        (
            stats_values,
            predictor_type,
            plot_paths,
            predictor_name,
            resp_name,
            resp_type,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        continuous_pred, categorical, boolean = self.checkColIsContOrCat()

        response_VarGroup = self.get_col_type(self.response)
        for cont_pred in continuous_pred:
            if continuous_pred is None:
                continue
            if response_VarGroup == "continuous":
                test = Cat_vs_Cont(
                    cont_pred,
                    response=self.response,
                    df=self.df,
                    predictors=self.predictors,
                ).contResponse_vs_contPredictor()
                _, _, _, file = test
                cont_name, t_value, p_value, _ = test
                stats_values.append(
                    (
                        cont_name,
                        t_value,
                        p_value,
                    )
                )
                predictor_name.append(cont_pred)
                predictor_type.append("Continuous")
                plot_paths.append(file)
                resp_name.append(self.response)
                resp_type.append(response_VarGroup)

            elif response_VarGroup == "categorical":
                test = Cat_vs_Cont(
                    cont_pred,
                    response=self.response,
                    df=self.df,
                    predictors=self.predictors,
                ).catResponse_vs_contPredictor()
                file = test
                predictor_name.append(cont_pred)
                predictor_type.append("Continuous")
                plot_paths.append(file)
                resp_name.append(self.response)
                resp_type.append(response_VarGroup)

            elif response_VarGroup == "boolean":
                test = Cat_vs_Cont(
                    cont_pred,
                    response=self.response,
                    df=self.df,
                    predictors=self.predictors,
                ).BoolResponse_vs_ContPredictor()
                cont_name, tval, pval, file = test
                stats_values.append((cont_name, tval, pval))
                predictor_name.append(cont_pred)
                predictor_type.append("Continuous")
                plot_paths.append(file)
                resp_name.append(self.response)
                resp_type.append(response_VarGroup)
            else:
                print(cont_pred, response_VarGroup)
                raise TypeError("invalid input...by me")

        # Plotting categorical predictors with response
        for cat_pred in categorical:
            if categorical is None:
                continue
            if response_VarGroup == "continuous":
                test = Cat_vs_Cont(
                    categorical=cat_pred,
                    response=self.response,
                    df=self.df,
                    predictors=self.predictors,
                ).contResponse_vs_catPredictor()
                file = test
                predictor_name.append(cat_pred)
                predictor_type.append("Categorical")
                plot_paths.append(file)
                resp_name.append(self.response)
                resp_type.append(response_VarGroup)

            elif response_VarGroup in ("categorical", "boolean"):
                test = Cat_vs_Cont(
                    categorical=cat_pred,
                    response=self.response,
                    df=self.df,
                    predictors=self.predictors,
                ).catResponse_vs_catPredictor()
                predictor_name.append(cat_pred)
                predictor_type.append("Categorical")
                plot_paths.append(file)
                resp_name.append(self.response)
                resp_type.append(response_VarGroup)
            else:
                print(cat_pred, response_VarGroup)
                raise AttributeError(
                    "Something is not being plotted correctly, issue with class?"
                )
        return (
            stats_values,
            predictor_type,
            plot_paths,
            predictor_name,
            resp_name,
            resp_type,
        )

    def plot_summary_html(self):
        (
            _,
            predictor_type,
            plot_paths,
            predictor_name,
            resp_name,
            resp_type,
        ) = self.predictor_plots()
        HW_4_html_df = pd.DataFrame(
            {
                "Predictor Type": predictor_type,
                "Predictor": predictor_name,
                "Response": resp_name,
                "Response type": resp_type,
                "Links to Plots": plot_paths,
            }
        )
        curr_path = self.get_workingDir()
        HW_4_html_df = HW_4_html_df.style.format({"Links to Plots": make_clickable})
        HW_4_html_df.to_html(
            f"{curr_path}/html_plots_and_tables/__HW4_plots.html", escape="html"
        )
        return
