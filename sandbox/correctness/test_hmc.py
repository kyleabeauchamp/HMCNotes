import os
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

precision = "mixed"

sysname = "longrfwater"

system, positions, groups, temperature, timestep, testsystem = lb_loader.load(sysname)

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, timestep / 4.)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(50000)
positions = context.getState(getPositions=True).getPositions()

collision_rate = 1.0 / u.picoseconds
n_steps = 25
Neff_cutoff = 1E5

grid = []
for itype in ["HMCIntegrator"]:
    d = dict(itype=itype, timestep=timestep)
    grid.append(d)


for settings in grid:
    itype = settings.pop("itype")
    settings["temperature"] = temperature
    timestep = settings["timestep"]
    if "RESPA" in itype:
        settings["groups"] = groups
    integrator = getattr(hmc_integrators, itype)(**settings)
    context = lb_loader.build(system, integrator, positions, temperature, precision=precision)
    filename = "./data/%s_%s_%s_%.3f_%d.csv" % (precision, sysname, itype, timestep / u.femtoseconds, collision_rate * u.picoseconds)
    print(filename)
    integrator.step(10000)
    data, start, g, Neff, mu, sigma, stderr = lb_loader.converge(context, n_steps=n_steps, Neff_cutoff=Neff_cutoff, filename=filename)
