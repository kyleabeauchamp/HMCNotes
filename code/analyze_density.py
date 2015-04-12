import pandas as pd
import pymbar
import numpy as np
import scipy.stats


reference = 0.90392903238929057
ref2fs = 0.9027456369177488


x = pd.read_csv("../production/production_2.00.log")

data = x["Density (g/mL)"]

n = len(data)
g = pymbar.timeseries.statisticalInefficiency(data)
neff = len(data) / g
mu = data.mean()
sigma = data.std()
stderr = sigma * neff ** -0.5

mu, reference, stderr
mu - reference
mu - ref2fs



x = pd.read_csv("./tables/timestep_dependence.csv")
power = 1.0
XY = x.set_index("timestep").ix[[0.5, 1.0]]
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(XY.index.values ** power, XY.mu.values)
f = lambda timestep: slope * timestep ** power + intercept
reference = intercept

