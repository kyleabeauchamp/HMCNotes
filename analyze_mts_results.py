import openmmtools
from simtk import unit as u
import pandas as pd
import glob

periods = 10

kT = openmmtools.constants.kB * 300. * u.kelvin / (u.kilojoules_per_mole)

key = "Total Energy (kJ/mole)"

filenames = glob.glob("./conservation/switchedaccurateflexiblewater*.csv")
#filenames = glob.glob("./oldconserv/switchedaccurateflexiblewater*750*.csv")
#filenames = glob.glob("./verlet/switchedaccurateflexiblewater*.csv")
#filenames = glob.glob("./conservation/switchedaccuratewater*.csv")

x = {}
for filename in filenames:
    xi = pd.read_csv(filename)
    x[filename] = xi[key]

x = pd.DataFrame(x)

delta = (x - x.iloc[0])
rho = (x - x.iloc[0]) / (x.iloc[0])

pval = np.exp(-x.diff(periods) / kT)
pval = pd.DataFrame(np.where(pval > 1, 1, pval), index=pval.index, columns=pval.columns)

print(abs(rho).mean())

print(x.diff().mean())

print((x.diff() ** 2).mean() ** 0.5)

print(pval.mean())
