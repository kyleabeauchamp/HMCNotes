import glob
import scipy.stats
import pymbar
import pandas as pd
import numpy as np
pd.set_option('display.width', 1000)

filenames = glob.glob("data/*.csv")
data = []
for filename in filenames:
    energies = pd.read_csv(filename)["Potential Energy (kJ/mole)"].values
    g = pymbar.timeseries.statisticalInefficiency(energies)
    Neff = len(energies) / g
    mu = energies.mean()
    sigma = energies.std()
    stderr = sigma * Neff ** -0.5

    data.append(dict(filename=filename, g=g, Neff=Neff, mu=mu, sigma=sigma, stderr=stderr))

data = pd.DataFrame(data)
data.sort("filename")
