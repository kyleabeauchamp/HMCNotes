import seaborn as sns
sns.set(style="whitegrid")
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
import collections
import itertools
import sklearn.grid_search
pd.set_option('display.width', 1000)

platform_name = "CUDA"
precision = "mixed"

#sysname = "switchedaccuratebigflexiblewater"
sysname = "switchedaccurateflexiblewater"
filename = "./data/gridsearch/%s.csv" % sysname

results = pd.read_csv(filename)
results = results.dropna()

results["RESPA"] = results.group0_iterations.astype('str') + "x" + results.groups.astype("str")

r = results[results.intname == "GHMCRESPAIntegrator"].groupby(["RESPA", "timestep"]).mean().reset_index()

for groupname, group in r.groupby("RESPA"):
    #sns.lmplot(x="timestep", y="effective_ns_per_day", hue="group0_iterations", data=group, ci=None, order=2)
    #plt.title(groupname)
    print(group)

sns.lmplot(x="timestep", y="effective_ns_per_day", hue="group0_iterations", col="RESPA", data=r, ci=None, order=2, col_wrap=3)
