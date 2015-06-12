import scipy.stats
import pymbar
import pandas as pd
import numpy as np
pd.set_option('display.width', 1000)

filename = "./data/mixed_chargedswitchedaccurateljbox_UnrolledXCHMCIntegrator_10.000_1.csv"

energies = pd.read_csv(filename)["Potential Energy (kJ/mole)"].values

g = pymbar.timeseries.statisticalInefficiency(energies)
Neff = len(energies) / g
mu = energies.mean()
sigma = energies.std()
stderr = sigma * Neff ** -0.5

print(mu, stderr)
print(mu + 27102.4)
