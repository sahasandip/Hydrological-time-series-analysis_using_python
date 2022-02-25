import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo
from my_config import *


class ForecastApp(tk.Frame):
    """
     Written by: Asheka
     step 1: Select daily average flow data csv file.
     step 2: click "compute" button to get the next 10 days water level forecast.
     step 3: From the combo box select a date
     step 4: click check button to show the respective water level.
     """

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master.title("ForecastApp")
        # setting up the geometry
        self.master.geometry("700x400")

        self.a_label = tk.Label(master,
                                text="Welcome To Forecast App",
                                font=1, foreground="green")
        self.a_label.grid(column=2, row=0,
                          pady=5, padx=5)

        self.b_label = tk.Label(master,
                                text="Step 1: Select you daily average CSV file",
                                foreground="green")
        self.b_label.grid(column=0, row=3,
                          padx=5, pady=5)

        # Button to select a csv file
        self.a_button = tk.Button(master, text="Select",
                                  foreground="blue",
                                  command=lambda: self.select_file())
        self.a_button.grid(column=0, row=4,
                           padx=5, pady=5)

        self.avg_file_info = None
        self.forecast_levels = None
        self.my_forecast_dates = None

        # add a compute label
        self.compute_label = tk.Label(master,
                                      text="Step 2: CLICK below to get the forecast data. ",
                                      foreground="green")
        self.compute_label.grid(column=2, row=4,
                                padx=5, pady=5)

        # add a compute button.
        self.compute_button = tk.Button(master,
                                        text="Compute", foreground="blue",
                                        command=lambda: self.get_forecasting())
        self.compute_button["state"] = "disabled"
        self.compute_button.grid(column=2, row=5,
                                 padx=5, pady=5)

        # Label to select a date.
        self.compute_label = tk.Label(master,
                                      text="Step 3:Select a date", foreground="green")
        self.compute_label.grid(column=0, row=6)

        # Adding a combo box to select the dates.
        self.cmb = ttk.Combobox(master)
        self.cmb["state"] = "disabled"
        self.cmb["values"] = [""]
        self.cmb.grid(column=0, row=7)

        # Button to check the respective water level
        self.combo_check = tk.Button(master,
                                     text="Check", foreground="blue",
                                     state="disabled", command=lambda: self.get_info())
        self.combo_check.grid(column=1, row=7)

        # Add a image
        my_img = PhotoImage(file="Hofkirchen.png").subsample(5, 5)
        self.myimg1 = tk.Label(master, image=my_img)
        self.myimg1.image = my_img
        self.myimg1.grid(row=9, column=2, rowspan=4)

    def select_file(self):
        """
        Open the local folder to select the file.
        disable the select button and enable compute button.
        """
        try:
            na = askopenfilename(initialdir="danube", filetypes=(("csv files", "*.csv"), ("All files", "*")))
            bs = na.rsplit("/", 1)
            ann = bs[0]
            ass = bs[1]
            avgavg = pkg.DataRead(ann, ass)
            self.avg_file_info = avgavg.get_dataframe()
            self.compute_button["state"] = ["normal"]
            self.a_button["state"] = "disabled"
            return self.avg_file_info
        except IndexError:
            showinfo("Error!", "NO file is chosen")

    def get_forecasting(self):
        """
        Return the water level data.
        Enable combo check button.
        """
        try:
            mmss = self.avg_file_info
            wwss = pkg.Modelling(mmss)
            self.forecast_levels = wwss.get_forecast_model()
            self.my_forecast_dates = pd.date_range(start=mmss["Date"].iloc[-1],
                                                   periods=10).strftime("%Y-%m-%d").tolist()
            self.cmb['state'] = 'readonly'
            self.cmb['values'] = self.my_forecast_dates
            self.combo_check["state"] = "normal"

            self.compute_button["state"] = "disabled"
            self.forecast_levels = pd.Series.tolist(self.forecast_levels)
            return self.forecast_levels

        except KeyError:
            showinfo("Error", "This is not the desired file")

    def get_info(self):
        """
        Return the result.
        """
        abww = self.my_forecast_dates
        qqqe = self.forecast_levels

        if self.cmb.get() == abww[0]:
            showinfo("Result", "Water level is  %s mÂ³/s" % str(qqqe[0]))
        elif self.cmb.get() == abww[1]:
            showinfo("Result", "Water level is  %s mÂ³/s" % str(qqqe[1]))
        elif self.cmb.get() == abww[2]:
            showinfo("Result", "Water level is  %s mÂ³/s" % str(qqqe[2]))
        elif self.cmb.get() == abww[3]:
            showinfo("Result", "Water level is  %s mÂ³/s" % str(qqqe[3]))
        elif self.cmb.get() == abww[4]:
            showinfo("Result", "Water level is  %s mÂ³/s" % str(qqqe[4]))
        elif self.cmb.get() == abww[5]:
            showinfo("Result", "Water level is  %s mÂ³/s" % str(qqqe[5]))
        elif self.cmb.get() == abww[6]:
            showinfo("Result", "Water level is  %s mÂ³/s" % str(qqqe[6]))
        elif self.cmb.get() == abww[7]:
            showinfo("Result", "Water level is  %s mÂ³/s" % str(qqqe[7]))
        elif self.cmb.get() == abww[8]:
            showinfo("Result", "Water level is  %s mÂ³/s" % str(qqqe[8]))
        elif self.cmb.get() == abww[9]:
            showinfo("Result", "Water level is  %s mÂ³/s" % str(qqqe[9]))
        else:
            showinfo("Action required", "Please select a date")


if __name__ == '__main__':
    ForecastApp().mainloop()
