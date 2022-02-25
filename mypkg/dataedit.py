import numpy as np
import pandas as pd


class DataRead:
    """
    Written by: Sandip.
    Read and clean csv file.
    Replace XXX value with Nan,
    Calculate the daily average for discrete time series data from a station.
    """

    def __init__(self, pathname: str,
                 file_name: str,
                 delimiter=","):
        """magic method.
        inputs: file pathname and file name"""
        self.pathname = pathname
        self.file_name = file_name
        self.sep = delimiter
        self.df = pd.DataFrame
        self.clean_df = pd.DataFrame

    def get_dataframe(self):
        """ Return a dataframe from csv file
        """
        self.df = pd.read_csv(self.pathname + '\\' + self.file_name,
                              header=0,
                              index_col=False,
                              skipinitialspace=True,
                              skip_blank_lines=True)
        return self.df

    def get_time_list(self):
        """
        Return the first column of panda data frame as time list
        """
        wr = self.get_dataframe()
        # Replace 24:00 to 23:59 to avoid time conflict.
        data_column_str = (wr.iloc[:, 0]).str.split(";",
                                                    expand=True).replace("24:00", "23:59")
        time_sequence_list = data_column_str.iloc[:, 0].values.tolist()

        return time_sequence_list

    def get_level_list(self):
        """
        Return the second row of dataframe after replacing XXX value with np.nan.
        Return as list.
        """
        wl = self.get_dataframe()
        data_column_st = (wl.iloc[:, 0]).str.split(";", expand=True)
        water_level_df = data_column_st.iloc[:, 1].str.replace('"', '')
        water_level_df = water_level_df.replace("XXX", np.nan, regex=True)
        w_level_df = pd.to_numeric(water_level_df)
        w_level_list = w_level_df.values.tolist()
        return w_level_list

    def get_clean_data(self):
        """
        Return a panda data frame with two list (time and water_level)
        """
        tt = self.get_time_list()
        ll = self.get_level_list()
        dictionary = dict(zip(tt, ll))
        self.clean_df = pd.DataFrame(dictionary.items(), columns=["Time", "Water_Level"])
        return self.clean_df

    @property
    def get_average_data(self):
        """
        Return average water level  for a day.
        """
        cdf = self.get_level_list()
        average = np.nanmean(cdf).round(2)
        return average


if __name__ == '__main__':
    try:
        data1 = DataRead("D:\pythonProject\danube\my_data", "01.12.2021.csv")
        print(data1.get_average_data)

    except FileNotFoundError:
        print("Please enter the correct file path")

    
