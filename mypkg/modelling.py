import numpy as np
import pandas as pd
import itertools
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


class Modelling:
    """
    Written by: Asheka
    Perform forecasting for time series with arima model.
    The order  of arima model  is selected for least root mean squared error
    """

    def __init__(self, my_frame):
        # magic method.
        self.my_frame = my_frame
        # p, d, q is selected for a wide range
        self.p = range(0, 4)
        self.d = range(0, 4)
        self.q = range(0, 4)
        self.orders = []
        self.rmse = []

    def get_train(self):
        # create a dataframe with first 50 value.
        b = self.my_frame
        train = b[:50]
        return train

    def get_test(self):
        # create a dataframe from 51 value to the end
        c = self.my_frame
        test = c[50:]
        return test

    def get_ordrmse(self):
        """
        Predict for train data with different set of p,d,q.
        errors are calculated between test and predicted data.
        return a dataframe with pdq set and their respective errors.
        """
        a = self.my_frame
        # create a list of p, d, q set for the given range
        pdq_list = list(itertools.product(self.p, self.d, self.q))
        # for each set of p, d, q value arima model is applied to the train data set.
        for elements in pdq_list:
            model = ARIMA(self.get_train()["Average_Water_Level"], order=elements).fit()
            model_fit = (model.predict(start=len(self.get_train()), end=len(a) - 1))
            # errors are calculated for each predicted model with the test data set.
            errors = np.sqrt(mean_squared_error(self.get_test()["Average_Water_Level"], model_fit))
            # Two list of orders and their respective model errors  are calculated.
            self.orders.append(elements)
            self.rmse.append(errors)

            # A data frame is created with list of orders and sorted rmse.
        b_dic = dict(zip(self.orders, self.rmse))
        ord_df = pd.DataFrame(b_dic.items(), columns=["orders", "rmse"])
        ord_rmse_frame = ord_df.sort_values("rmse", ascending=True, ignore_index=True)
        return ord_rmse_frame

    def get_my_pdq(self):
        """ from a dataframe of pdq set and their respective errors, it
        chooses the order which gives the least error.
        """
        try:
            my_pdq = self.get_ordrmse().loc[0].at["orders"]
        except KeyError:
            print("no such dataframe with column name")
        return my_pdq

    def get_my_rmse(self):
        """ return rmse for choosen pdq"""
        error = self.get_ordrmse().loc[0].at["rmse"]
        return error

    def get_pred_model(self):
        """
        Return prediction for train data set with a length of test data set.
        """
        try:
            pred_model = ARIMA(self.get_train()["Average_Water_Level"], order=self.get_my_pdq()).fit()
            pred_fit = pred_model.predict(start=len(self.get_train()), end=len(self.my_frame) - 1)
        except KeyError:
            print("no such list with column name")
        return pred_fit

    def get_my_r2(self):
        """
        Return correlation co efficient for predicted model and test data set.
        """
        x = pd.to_numeric(self.get_test()["Average_Water_Level"])
        y = pd.to_numeric(self.get_pred_model())
        correlation_coefficient = x.corr(y)
        return correlation_coefficient

    def get_forecast_model(self):
        """
        Return prediction for total timeseries set with the chosen order of pdq for least error.
        The length of prediction is 10.
        """

        final_model = ARIMA(self.my_frame["Average_Water_Level"], order=self.get_my_pdq()).fit()
        final_fit = (final_model.predict(start=len(self.my_frame), end=len(self.my_frame) + 9))
        return final_fit

    def __str__(self):
        """ Magic method to print the errors"""
        return f"the root mean square error of our model is {self.get_my_rmse()}\n " \
               f"the correlation coefficient is {self.get_my_r2()}"


if __name__ == '__main__':
    try:
        my_df = pd.read_csv("D:\pythonProject\danube\modified_average_file.csv", header=0,
                            index_col=False,
                            skipinitialspace=True,
                            skip_blank_lines=True)
        datas = Modelling(my_df)
        print(datas.get_forecast_model())
    except FileNotFoundError:
        print("Please enter the correct file path")

