import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems
pd.set_option('display.width', 1000)

collision_rate = 10000.0 / u.picoseconds

sysname = "density"

system, positions, groups, temperature, timestep = lb_loader.load(sysname)

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, timestep / 4.)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(20000)
positions = context.getState(getPositions=True).getPositions()

Neff_cutoff = 2000.

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, timestep / 2.)
context = lb_loader.build(system, integrator, positions, temperature)
data, start, g, Neff = lb_loader.converge(context, n_steps=100, Neff_cutoff=Neff_cutoff)
data.to_csv("./data/%s_langevin.csv" % sysname)

integrator = integrators.GHMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=collision_rate)
context = lb_loader.build(system, integrator, positions, temperature)
data, start, g, Neff = lb_loader.converge(context, n_steps=10, Neff_cutoff=Neff_cutoff)
data.to_csv("./data/%s_ghmc.csv" % sysname)

integrator = integrators.GHMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds)
context = lb_loader.build(system, integrator, positions, temperature)
data, start, g, Neff = lb_loader.converge(context, n_steps=10, Neff_cutoff=Neff_cutoff)
data.to_csv("./data/%s_ghmc1.csv" % sysname)

integrator = integrators.XHMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds, extra_chances=5)
context = lb_loader.build(system, integrator, positions, temperature)
data, start, g, Neff = lb_loader.converge(context, n_steps=10, Neff_cutoff=Neff_cutoff)
data.to_csv("./data/%s_xhmc1.csv" % sysname)

integrator = integrators.GHMCRESPA(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds, groups=groups)
context = lb_loader.build(system, integrator, positions, temperature)
data, start, g, Neff = lb_loader.converge(context, n_steps=10, Neff_cutoff=Neff_cutoff)
data.to_csv("./data/%s_rghmc1.csv" % sysname)

integrator = integrators.XHMCRESPAIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds, extra_chances=5, groups=groups)
context = lb_loader.build(system, integrator, positions, temperature)
data, start, g, Neff = lb_loader.converge(context, n_steps=10, Neff_cutoff=Neff_cutoff)
data.to_csv("./data/%s_rxhmc1.csv" % sysname)
