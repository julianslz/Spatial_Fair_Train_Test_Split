import os

import numpy as np
import pandas as pd
from geostatspy.GSLIB import Dataframe2GSLIB, GSLIB2ndarray, DataFrame2ndarray


def kb2d(df, xcol, ycol, vcol, nx, ny, xmn, ymn, hsiz, var, sk_mean, output_file):
    """Kriging estimation, 2D wrapper for kb2d from GSLIB (.exe must be
    available in PATH or working directory).
    """
    df_temp = pd.DataFrame({"X": df[xcol], "Y": df[ycol], "Var": df[vcol]})
    Dataframe2GSLIB("data_temp.dat", df_temp)

    nug = var.get("nug")
    nst = var.get("nst")
    it1 = var.get("it1")
    cc1 = var.get("cc1")
    azi1 = var.get("azi1")
    hmaj1 = var.get("hmaj1")
    hmin1 = var.get("hmin1")
    it2 = var.get("it2")
    cc2 = var.get("cc2")
    azi2 = var.get("azi2")
    hmaj2 = var.get("hmaj2")
    hmin2 = var.get("hmin2")
    max_range = max(hmaj1, hmaj2)

    with open("kb2d.par", "w") as f:
        f.write("              Parameters for KB2D                                          \n")
        f.write("              ********************                                         \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETER:                                                        \n")
        f.write("data_temp.dat                         -file with data                      \n")
        f.write("1  2  3                               -  columns for X,Y,vr                \n")
        f.write("-1.0e21   1.0e21                      -   trimming limits                  \n")
        f.write("0                                     -debugging level: 0,1,2,3            \n")
        f.write("none.dbg                              -file for debugging output           \n")
        f.write(str(output_file) + "                   -file for kriged output              \n")
        f.write(str(nx) + " " + str(xmn) + " " + str(hsiz) + "                              \n")
        f.write(str(ny) + " " + str(ymn) + " " + str(hsiz) + "                              \n")
        f.write("1    1                                -x and y block discretization        \n")
        f.write("1    30                               -min and max data for kriging        \n")
        f.write(str(max_range) + "                     -maximum search radius               \n")
        f.write("0    " + str(sk_mean) + "              -0=SK, 1=OK,  (mean if SK)           \n")
        f.write(str(nst) + " " + str(nug) + "          -nst, nugget effect                  \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + " " + str(hmaj1) + " " + str(
            hmin1) + " -it, c ,azm ,a_max ,a_min \n")
        f.write(str(it2) + " " + str(cc2) + " " + str(azi2) + " " + str(hmaj2) + " " + str(
            hmin2) + " -it, c ,azm ,a_max ,a_min \n")

    os.system("kb2d.exe kb2d.par")
    est_array = GSLIB2ndarray(output_file, 0, nx, ny)
    var_array = GSLIB2ndarray(output_file, 1, nx, ny)
    return est_array[0], var_array[0]


class FairSplit:
    def __init__(self, df, xcoor, ycoor, feature, cell_size):

        dataset = df.copy()
        self.xcoor = xcoor
        self.ycoor = ycoor
        self.feature = feature
        self.cell_size = cell_size

        self.xmin, self.xmax, self.ymin, self.ymax = self._max_and_min(dataset)
        xi = (np.floor((dataset[self.xcoor] - self.xmin) / self.cell_size)).astype(int)
        yi = (np.floor((dataset[self.ycoor] - self.ymin) / self.cell_size)).astype(int)
        ncells = int((self.ymax - self.ymin) / self.cell_size)
        dataset['Cell x'] = xi
        dataset['Cell y'] = yi
        dataset['Cell id'] = ncells * yi + xi
        self.dataset = dataset

        self._nx = ncells
        self._ny = ncells
        self._xmn = self.xmin + self.cell_size / 2
        self._ymn = self.ymin + self.cell_size / 2

    def _max_and_min(self, df):
        """
        Compute the maximum and minimum values of a rectangle for modeling
        """
        # get the range in each dimension
        range_x = np.round(np.ptp(df[self.xcoor]))
        range_y = np.round(np.ptp(df[self.ycoor]))
        half = np.abs(range_x - range_y) / 2
        # add additional length to the smaller axis so the cell size is a multiple of the total length
        if range_x > range_y:
            ymin = np.round(df[self.ycoor].min()) - half
            ymax = np.round(df[self.ycoor].max()) + half
            xmax = np.round(df[self.xcoor].max())
            xmin = np.round(df[self.xcoor].min())
        else:
            xmin = np.round(df[self.xcoor].min()) - half
            xmax = np.round(df[self.xcoor].max()) + half
            ymax = np.round(df[self.ycoor].max())
            ymin = np.round(df[self.ycoor].min())

        if (xmax - xmin) % self.cell_size > 0:  # the cell size is not exact
            cells = int(np.ceil((xmax - xmin) / self.cell_size))
            half = (cells * self.cell_size - (xmax - xmin)) / 2
            ymin -= half
            ymax += half
            xmax += half
            xmin -= half

        return xmin, xmax, ymin, ymax

    def s_kriging(self, vario_model, output_file, sk_mean=None):
        if sk_mean is None:
            sk_mean = self.dataset[self.feature].mean()

        kmap, vmap = kb2d(
            self.dataset, self.xcoor, self.ycoor, self.feature, self._nx, self._ny, self._xmn, self._ymn,
            self.cell_size, vario_model, sk_mean, output_file
        )

        return kmap, vmap
