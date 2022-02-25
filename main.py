from my_config import *
warnings.filterwarnings("ignore")


def get_average_df():
    """ From  the path names created in "csv_files"
    Create a new file "modified_average_file.csv" with daily average data.
    Return the dataframe "myy_df" with the daily average value.
    """
    for f in csv_files:
        # split the file name from the csv files
        i = [f.split("\\")[-1]]
        # path name and file name is plugged in to DataRead class
        data1 = pkg.DataRead("my_data", i[-1])
        average_list.append(data1.get_average_data)
        date_list.append((f.split("\\")[-1].replace(".csv", "")))
        a_dic = dict(zip(date_list, average_list))
        average_df = pd.DataFrame(a_dic.items(), columns=["Date", "Average_Water_Level"])
        average_df["Date"] = pd.to_datetime(average_df["Date"], format="%d.%m.%Y")
        myy_df = average_df.sort_values("Date", ascending=True).reset_index(drop=True)
        try:
            new_file = myy_df.to_csv("modified_average_file.csv",
                                     index=False, header=True,
                                     mode="w+", sep=",")
        except PermissionError:
            print("Please close the file from the window")

    return myy_df


def my_forecast():
    """
    use the dataframe from the function "get_average_df" and
    return  the future water level values as "forecast_df"
    """
    frame = get_average_df()
    data_g = pkg.Modelling(frame)
    my_forecast_levels = data_g.get_forecast_model()
    datelist = pd.date_range(start=frame["Date"].iloc[-1], periods=10).tolist()
    my_dic = dict(zip(datelist, my_forecast_levels))
    forecast_df = pd.DataFrame(my_dic.items(), columns=["Date", "Estimated_Water_Level"])
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"], format="%d.%m.%Y")
    print(data_g)  # will print the errors
    return forecast_df


def forecast_plots():
    """
    Create a run plot where
    train = plotting of 50 days data
    actual = plotting of the rest days data
    my model = our model predicted with train data set for evaluation
    my forecast = our model prediction for total days
    """
    frame2 = get_average_df()
    mo = pkg.modelling.Modelling(frame2)
    datelist = pd.date_range(start=frame2["Date"].iloc[-1], periods=10).tolist()
    plt.plot(mo.get_train()["Date"], mo.get_train()["Average_Water_Level"], label="train")
    plt.plot(mo.get_test()["Date"], mo.get_test()["Average_Water_Level"], label="actual")
    plt.plot(mo.get_test()["Date"], mo.get_pred_model(), label="my model")
    plt.plot(datelist, mo.get_forecast_model(), label="my_forecast")
    plt.legend(loc='upper left', fontsize=8)
    plt.title("Daily_Water_Level_graph")
    plt.show()


if __name__ == '__main__':
    csv_files = glob.glob(os.path.join("my_data", "*.csv"))  # create a list of pathname of my_data folder
    average_list = []
    date_list = []
    print(my_forecast())
    forecast_plots()
