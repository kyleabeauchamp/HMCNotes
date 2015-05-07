import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

sysname = "ljbox"
collision_rate = 1.0 / u.picoseconds

system, positions, groups, temperature, timestep = lb_loader.load(sysname)

integrator = integrators.AndersenVelocityVerletIntegrator(temperature, collision_rate, timestep / 4.)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(50000)
positions = context.getState(getPositions=True).getPositions()

n_steps = 25
Neff_cutoff = 2000.


grid = []

for itype in ["AndersenVelocityVerletIntegrator"]:
    for timestep_factor in [1.0, 2.0, 4.0]:
        d = dict(itype=itype, timestep=timestep / timestep_factor)
        grid.append(d)


for settings in grid:
    itype = settings.pop("itype")
    timestep = settings["timestep"]
    integrator = integrators.AndersenVelocityVerletIntegrator(temperature, collision_rate, timestep / 4.)
    context = lb_loader.build(system, integrator, positions, temperature)
    filename = "./data/%s_%s_%.3f_%d.csv" % (sysname, itype, timestep / u.femtoseconds, collision_rate * u.picoseconds)
    print(filename)
    data, start, g, Neff = lb_loader.converge(context, n_steps=n_steps, Neff_cutoff=Neff_cutoff)
    data.to_csv(filename)
    
