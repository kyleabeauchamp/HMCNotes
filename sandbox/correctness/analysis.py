import scipy.stats
import pymbar
import pandas as pd
import numpy as np
import glob
pd.set_option('display.width', 1000)

filenames = glob.glob("./data/single/*.csv")

data = []
for filename in filenames:
    energies = pd.read_csv(filename).energy
    sysname, integrator, timestep, friction = filename[:-4].split("/")[-1].split(",")[0].split("_")
    (start, g, Neff) = pymbar.timeseries.detectEquilibration(energies)
    energies = energies[start:]
    mu = energies.mean()
    sigma = energies.std()
    stderr = sigma * Neff ** -0.5
    data.append(dict(sysname=sysname, integrator=integrator, timestep=float(timestep), friction=friction, start=start, g=g, Neff=Neff, sigma=sigma, stderr=stderr, mu=mu))
    print(data[-1])

data = pd.DataFrame(data)
data = data.sort("timestep")[::-1].sort("integrator")[::-1]
data["true"] = 0.0

for sysname, x in data.groupby("sysname"):
    a = x[x.integrator == "LangevinIntegrator"].timestep.values
    ind = a.argsort()
    b = x[x.integrator == "LangevinIntegrator"].mu.values
    a = a[ind][0:2]
    b = b[ind][0:2]
    slope, intercept, _, _, _ = scipy.stats.linregress(a, b)
    data["true"][data.sysname == sysname] = intercept

data["true"] = true
data["error"] = data.mu - data.true
data["relerror"] = data.error / data.true

X = data[data.sysname == "ho"]
X



import openmmtools
from simtk import unit as u
testsystem = openmmtools.testsystems.HarmonicOscillatorArray()
tstate = openmmtools.testsystems.ThermodynamicState(temperature=300*u.kelvin)
true = testsystem.get_potential_expectation(tstate) / u.kilojoules_per_mole

