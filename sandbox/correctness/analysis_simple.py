import glob
import scipy.stats
import pymbar
import pandas as pd
import numpy as np
pd.set_option('display.width', 1000)

def summarize(filename):
    statedata = pd.read_csv(filename)
    energies = statedata["Potential Energy (kJ/mole)"].values
    if "Kinetic Energy (kJ/mole)" in statedata.columns:
        holding = pd.read_csv(filename)["Kinetic Energy (kJ/mole)"].values
        mu = holding.dot(energies) / holding.sum()
        mu = energies.mean()
    else:
        mu = energies.mean()
    g = pymbar.timeseries.statisticalInefficiency(energies)
    Neff = max(1, len(energies) / g)
    sigma = energies.std()
    stderr = sigma * Neff ** -0.5

    return dict(filename=filename, g=g, Neff=Neff, mu=mu, sigma=sigma, stderr=stderr)

filenames = glob.glob("data/*.csv")
data = []
for filename in filenames:
    data.append(summarize(filename))

data = pd.DataFrame(data)
data.sort("filename")
