import itertools
import pandas as pd
import lb_loader
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

temperature = 300. * u.kelvin

testsystem = testsystems.LennardJonesCluster(2, 2, 2)

system, positions = testsystem.system, testsystem.positions
integrators.guess_force_groups(system, nonbonded=0, fft=0, others=0)

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.25 * u.femtoseconds)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)
integrator.step(75000)
positions = context.getState(getPositions=True).getPositions()


def sample(context, n_iter=10, n_steps=1):
    integrator = context.getIntegrator()
    data = []
    for i in itertools.count():
        integrator.step(n_steps)
        if type(integrator) in ["XHMCIntegrator", "XHMCRESPAIntegrator"]:
            if integrator.getGlobalVariableByName("a") != 1.:
                continue
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy() / u.kilojoules_per_mole
        data.append(dict(energy=energy))
        if len(data) >= n_iter:
            return data


n_iter = 3200
data = {}

collision_rate = 10000.0 / u.picoseconds
timestep = 90.0 * u.femtoseconds

integrator = integrators.GHMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=collision_rate)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)


data["ghmc"] = pd.DataFrame(sample(context, n_iter=n_iter, n_steps=1))


data = pd.Panel(data)

data.mean(1)
data.iloc[:, :, 0].plot()
