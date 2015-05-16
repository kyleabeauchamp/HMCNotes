import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

actual_timestep = 1.5 * u.femtoseconds

data = []
for sysname in ["ljbox", "switchedljbox", "shiftedljbox", "shortwater", "shortswitchedwater"]:
    system, positions, groups, temperature, timestep, langevin_timestep, testsystem = lb_loader.load(sysname)
    timestep = actual_timestep

    integrator = hmc_integrators.HMCIntegrator(temperature, steps_per_hmc=25, timestep=timestep)
    context = lb_loader.build(system, integrator, positions, temperature)
    mm.LocalEnergyMinimizer.minimize(context)
    integrator.step(500)
    positions = context.getState(getPositions=True).getPositions()

    integrator = hmc_integrators.HMCIntegrator(temperature, steps_per_hmc=25, timestep=timestep)
    context = lb_loader.build(system, integrator, positions, temperature)
    integrator.step(20000)
    print(integrator.acceptance_rate)

    data.append(dict(sysname=sysname, accept=integrator.acceptance_rate))

data = pd.DataFrame(data)
