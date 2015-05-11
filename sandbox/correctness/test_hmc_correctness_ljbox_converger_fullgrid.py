import os
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

precision = "single"

sysname = "ho"

system, positions, groups, temperature, timestep = lb_loader.load(sysname)

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, timestep / 4.)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(50000)
positions = context.getState(getPositions=True).getPositions()

collision_rate = 1.0 / u.picoseconds
n_steps = 25
Neff_cutoff = 6000.

# HACK to facilitate iterating over hmc_integrators
def LangevinIntegrator(temperature=None, timestep=None):
    return mm.LangevinIntegrator(temperature, collision_rate, timestep)

hmc_integrators.LangevinIntegrator = LangevinIntegrator

grid = []

#for itype in ["LangevinIntegrator", "HMCIntegrator", "GHMCIntegrator", "XCHMCIntegrator", "XCGHMCIntegrator", "HMCRESPAIntegrator", "GHMCRESPAIntegrator", "XCHMCRESPAIntegrator", "XCGHMCRESPAIntegrator"]:
for itype in ["LangevinIntegrator", "HMCIntegrator", "GHMCIntegrator"]:
    for timestep_factor in [1.0, 2.0, 4.0, 8.0]:
        d = dict(itype=itype, timestep=timestep / timestep_factor)
        grid.append(d)



for settings in grid:
    itype = settings.pop("itype")
    settings["temperature"] = temperature
    timestep = settings["timestep"]
    if "RESPA" in itype:
        settings["groups"] = groups
    integrator = getattr(hmc_integrators, itype)(**settings)
    context = lb_loader.build(system, integrator, positions, temperature, precision=precision)
    filename = "./data/%s/%s_%s_%.3f_%d.csv" % (precision, sysname, itype, timestep / u.femtoseconds, collision_rate * u.picoseconds)
    if os.path.exists(filename):
        continue
    print(filename)
    data, start, g, Neff = lb_loader.converge(context, n_steps=n_steps, Neff_cutoff=Neff_cutoff)
    data.to_csv(filename)
    
