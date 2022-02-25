from my_config import *

warnings.filterwarnings("ignore")

my_df = pd.read_csv("modified_average_file.csv", header=0,
                    index_col=False,
                    skipinitialspace=True,
                    skip_blank_lines=True)


def get_d():
    """ create auto-correlation plot for timeseries.
    differentiation order (d) = 0
    d = 1
    d= 2
    After observing the plot, d is entered manually
    """

    a = data3.get_get_timeseries()
    # Perform stationary check.
    print(data3.adf_test(a))

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(a)
    axes[0, 0].set_title('my actual series')

    # Auto correlation (acf) plot for d=0
    plot_acf(a, ax=axes[0, 1])
    axes[0, 1].set_title("Acf plot for d = 0")

    # acf plot for d = 1
    plot_acf(a.diff().dropna(), ax=axes[1, 0])
    axes[1, 0].set_title("Acf plot for d = 1")

    # acf plot for d = 2
    plot_acf(a.diff().diff().dropna(), ax=axes[1, 1])
    axes[1, 1].set_title("Acf plot for d = 2")
    
    plt.subplots_adjust(wspace=0.3,
                        hspace=0.3)

    plt.show()
    ba = input("Enter d:")
    return ba


def get_my_order():
    """
     For the choosen order of d acf and pacf plot is created
     After visualization of acf and pacf plot, p and q value need
     to be entered manually.
     return p, d, q as tuple.
     """
    ww = data3.get_get_timeseries()
    # differentiation order.
    i = get_d()
    m =".diff()"
    s = (m * int(i))

    # This string contains the command.
    # if i (d) = 2; string will be: "plot_acf(ww.diff().diff().dropna())"
    code = ("plot_acf(ww{}.dropna())".format(s))

    # This string contains the command.
    # if i (d) = 2; string will be: "plot_pacf(ww.diff().diff().dropna())"
    mm = ("plot_pacf(ww{}.dropna())".format(s))

    exec(code)
    exec(mm)
    plt.show()

    # Need to enter p and q value from the prompt.
    my_p = input("Enter p:")
    my_q = input("Enter q:")
    my_order = int(my_p), int(i), int(my_q)

    return my_order


def my_forecast():
    """
    Return forecast for next 10 days.
    """
    my_tuple = get_my_order()
    data3.my_order = my_tuple
    result = data3.my_forecast()
    print(data3())
    print("Next 10 days water level for Hofkirchen station")
    print(result)
    return result


if __name__ == '__main__':
    data3 = pkg.Classical(my_df)
    my_forecast()

