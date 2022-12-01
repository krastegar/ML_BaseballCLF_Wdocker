from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split

from midterm import BruteForce, read_data


class TrainTestModel(BruteForce, read_data):
    """
    For doing training on sql baseball data we have to
    keep local_date=1 for non_shuffle split that way we can
    take all the old games and predict them on the new ones
    """

    def __init__(self, local_date=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_date = local_date

    def cat_transform(self):
        for col_name in list(self.df.columns):
            pred_type = self.get_col_type(col_name)
            if pred_type == "categorical":
                self.df[col_name], _, _ = self.cat_pred_bin(self.df[col_name])
        return self.df

    def sort_split(self):
        self.df = self.cat_transform()
        if self.local_date == 1:
            # shuffle needs to be false for this method
            df = self.df.sort_values(
                "local_date", ascending=True
            )  # hard coded for game_table in sql
            X, y = df[self.predictors], df[self.response]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            return X_train, X_test, y_train, y_test
        else:
            X, y = self.df[self.predictors], self.df[self.response]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test

    def trainer_tester(self):
        X_train, X_test, y_train, y_test = self.sort_split()
        try:
            model_1 = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ).fit(X_train, y_train)

            model_2 = LogisticRegression(penalty="l2", random_state=42).fit(
                X_train, y_train
            )
            y_pred_model_1, y_pred_model_2 = model_1.predict(X_test), model_2.predict(
                X_test
            )
            cm_1, cm_2 = confusion_matrix(y_test, y_pred_model_1), confusion_matrix(
                y_test, y_pred_model_2
            )
            rep_1 = classification_report(y_true=y_test, y_pred=y_pred_model_1)
            rep_2 = classification_report(y_true=y_test, y_pred=y_pred_model_2)
            print("GradientBoostClassifier report: \n", rep_1)
            print("LogisticRegression report: \n", rep_2)
            return cm_1, cm_2
        except ValueError:
            model_1 = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ).fit(X_train, y_train)

            model_2 = LinearRegression().fit(X_train, y_train)
            y_pred_model_1, y_pred_model_2 = model_1.predict(X_test), model_2.predict(
                X_test
            )
            m_1, m_2 = mean_squared_error(y_test, y_pred_model_1), mean_squared_error(
                y_test, y_pred_model_2
            )
            rep_1 = f"Mean Squared Error of GradientBoostingRegressor: {m_1}"
            rep_2 = f"Mean Squared Error of LinearRegressor: {m_2}"
            print(rep_1, "\n", rep_2)
            return m_1, m_2
