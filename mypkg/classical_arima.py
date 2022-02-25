import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller


class Classical:
    """
    Written by: Asheka
    This class contains methods  for time series forecasting with arima model,
    where the parameters (p,d,q) need to input manually.
    """

    def __init__(self, frame):
        self.frame = frame
        self._order = 0, 0, 0

    def get_get_timeseries(self):

        """ From a dataframe column with "Average_Water_Level" is selected as timeseries"""

        a = self.frame
        my_timeseries = a["Average_Water_Level"]
        return my_timeseries

    def adf_test(self, timeseries):

        """Perform Dickey-Fuller test to check if the data is stationary or not"""

        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries)
        print('p-value: %f' % dftest[1])
        print("ADF statistics:%f" % dftest[0])
        if dftest[1] <= 0.05:
            print("p value less than 0.05. Data is stationary")
        else:
            print("p value greater than 0.05.Data is non stationary")
        return 0

    @property  # property decorator
    def my_order(self):
        return self._order

    @my_order.setter
    def my_order(self, new_order):

        """ order need to be in tuple or in string """

        if isinstance(new_order, tuple):
            self._order = new_order
        elif isinstance(new_order, str):
            self._order = tuple(map(int, new_order.split(',')))
        else:
            print("enter a valid order")
        return new_order

    def my_train_model(self):
        """ predict for train data set"""

        train = self.get_get_timeseries()[:50]
        try:
            pred_model = ARIMA(train, order=self.my_order).fit()
            pred_fit = pred_model.predict(start=len(train), end=len(self.frame) - 1)
        except KeyError:
            print("no such list with column name")
        return pred_fit

    def my_forecast(self):
        """ Predict data for a time series"""
        final_model = ARIMA(self.get_get_timeseries(), order=self.my_order).fit()
        final_fit = (final_model.predict(start=len(self.frame), end=len(self.frame) + 9))
        return final_fit

    def __call__(self):
        """ Magic method to print the root mean squared error, and co-relation coefficient"""

        test = self.get_get_timeseries()[50:]
        errors = np.sqrt(mean_squared_error(test, self.my_train_model()))
        rsq = test.corr(self.my_train_model())
        print("my chosen order (p, d, q) for ARIMA model is {} ".format(self.my_order))
        print("model error(rmse) is {} and correlation coefficient is {}".format(errors, rsq))
        return -1


if __name__ == '__main__':
    try:
        my_df = pd.read_csv("D:\pythonProject\danube\modified_average_file.csv", header=0,
                            index_col=False,
                            skipinitialspace=True,
                            skip_blank_lines=True)
        datas = Classical(my_df)
        datas.my_order = "1 ,0 ,1"
        print(datas.get_get_timeseries())
        datas()
    except FileNotFoundError:
        print("Please enter the correct file path")
