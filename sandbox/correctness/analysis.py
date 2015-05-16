import collections
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
import lb_loader
import scipy.stats
import pymbar
import pandas as pd
import numpy as np
import glob
pd.set_option('display.width', 1000)

#system, positions, groups, temperature, timestep, langevin_timestep, testsystem = lb_loader.load("customho")
#E0 = (3/2.) * testsystem.n_particles * testsystems.kB * temperature / u.kilojoules_per_mole
true = collections.defaultdict(lambda: np.nan)
#true["customho"] = E0

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
    data.append(dict(precision=precision, sysname=sysname, integrator=integrator, timestep=float(timestep), friction=friction, start=start, g=g, Neff=Neff, sigma=sigma, stderr=stderr, mu=mu, samples=energies[::int(round(g))]))
    print(data[-1])

data = pd.DataFrame(data)
data = data.sort("timestep")[::-1].sort("integrator")[::-1].sort("precision").sort("sysname")
data["true"] = data.sysname.map(lambda x: true[x])


for (precision, integrator, sysname), x in data.groupby(("precision", "integrator", "sysname")):
    if integrator != "LangevinIntegrator":
        continue
    a = x.timestep.values
    ind = a.argsort()
    b = x.mu.values
    a = a[ind][0:2]
    b = b[ind][0:2]
    slope, intercept, _, _, _ = scipy.stats.linregress(a, b)

data["error"] = data.mu - data.true
data["relerror"] = data.error / data.true
data["pval"] = np.nan
#data[["sysname", "integrator", "mu", "stderr", "Neff", "timestep"]]
data = data.drop(["start", "friction"], axis=1)

for (precision, sysname), di in data.groupby(["precision", "sysname"]):
    x = di.set_index("integrator").ix["LangevinIntegrator"].samples
    y = di.set_index("integrator").ix["HMCIntegrator"].samples
    delta = di.set_index("integrator").ix["LangevinIntegrator"].mu - di.set_index("integrator").ix["HMCIntegrator"].mu
    relerr = delta / di.set_index("integrator").ix["HMCIntegrator"].mu
    tscore, pval = scipy.stats.ttest_ind(x, y, equal_var=False)
    data.loc[(data.precision == precision) & (data.sysname == sysname), "tscore"] = tscore
    data.loc[(data.precision == precision) & (data.sysname == sysname), "pval"] = pval
    data.loc[(data.precision == precision) & (data.sysname == sysname), "relerr"] = relerr
    data.loc[(data.precision == precision) & (data.sysname == sysname), "error"] = delta


data.drop("samples", axis=1)  # Printing the samples looks bad.


data[data.sysname.isin(["ljbox", "switchedljbox", "shiftedljbox"])].drop("samples", axis=1)  # Printing the samples looks bad.
