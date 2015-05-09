from simtk import unit as u
from openmmtools import integrators, testsystems
import lb_loader
import scipy.stats
import pymbar
import pandas as pd
import numpy as np
import glob
pd.set_option('display.width', 1000)

system, positions, groups, temperature, timestep, testsystem = lb_loader.load("customho")
E0 = (3/2.) * testsystem.n_particles * testsystems.kB * temperature / u.kilojoules_per_mole
true = {"customho":E0}

filenames = glob.glob("./data/*.csv")

data = []
for filename in filenames:
    energies = pd.read_csv(filename).energy
    precision, sysname, integrator, timestep, friction = filename[:-4].split("/")[-1].split(",")[0].split("_")
    start = 0
    g = pymbar.timeseries.statisticalInefficiency(energies)
    Neff = len(energies) / g
    energies = energies[start:]
    mu = energies.mean()
    sigma = energies.std()
    stderr = sigma * Neff ** -0.5
    data.append(dict(precision=precision, sysname=sysname, integrator=integrator, timestep=float(timestep), friction=friction, start=start, g=g, Neff=Neff, sigma=sigma, stderr=stderr, mu=mu))
    print(data[-1])

data = pd.DataFrame(data)
data = data.sort("timestep")[::-1].sort("integrator")[::-1].sort("precision")
data["true"] = data.sysname.map(lambda x: true[x])
data["extrapolated"] = data.mu


for (precision, integrator, sysname), x in data.groupby(("precision", "integrator", "sysname")):
    if integrator != "LangevinIntegrator":
        continue
    a = x.timestep.values
    ind = a.argsort()
    b = x.mu.values
    a = a[ind][0:2]
    b = b[ind][0:2]
    slope, intercept, _, _, _ = scipy.stats.linregress(a, b)
    data["extrapolated"][(data.sysname == sysname) & (data.precision == precision) & (data.integrator == integrator)] = intercept

data["error"] = data.mu - data.true
data["relerror"] = data.error / data.true
data["zscore"] = (data.mu - data.true) / data.stderr
data
