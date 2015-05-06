import pymbar
import pandas as pd
import numpy as np
import glob

filenames = glob.glob("./data/*.csv")

data = []
for filename in filenames:
    energies = pd.read_csv(filename).energy
    sysname, integrator, timestep, friction = filename[:-4].split("/")[-1].split(",")[0].split("_")
    (start, g, Neff) = pymbar.timeseries.detectEquilibration(energies)
    energies = energies[start:]
    mu = energies.mean()
    sigma = energies.std()
    stderr = sigma * Neff ** -0.5
    data.append(dict(sysname=sysname, integrator=integrator, timestep=timestep, friction=friction, start=start, g=g, Neff=Neff, sigma=sigma, stderr=stderr, mu=mu))
    print(data[-1])

data = pd.DataFrame(data)


x = data[data.sysname == "ljbox"]
