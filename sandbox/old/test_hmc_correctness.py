import itertools
import pandas as pd
import lb_loader
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

collision_rate = 10000.0 / u.picoseconds
temperature = 300. * u.kelvin

testsystem = testsystems.LennardJonesCluster(2, 2, 2)
#testsystem = testsystems.LennardJonesGrid()
#testsystem = testsystems.HarmonicOscillator()
#testsystem = testsystems.SodiumChlorideCrystal()
#testsystem = testsystems.WaterBox(box_edge=2.25 * u.nanometers)

system, positions = testsystem.system, testsystem.positions
hmc_integrators.guess_force_groups(system, nonbonded=0, fft=0, others=0)

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


timestep = 75 * u.femtoseconds  # LJ Cluster
n_iter = 100000
data = {}

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, timestep)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

data["langevin"] = pd.DataFrame(sample(context, n_iter=n_iter, n_steps=20))


integrator = hmc_integrators.GHMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=collision_rate)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)


data["ghmc"] = pd.DataFrame(sample(context, n_iter=n_iter, n_steps=1))


integrator = hmc_integrators.GHMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)


data["ghmc1"] = pd.DataFrame(sample(context, n_iter=n_iter, n_steps=1))


integrator = hmc_integrators.XHMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds, extra_chances=5)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

data["xhmc1"] = pd.DataFrame(sample(context, n_iter=n_iter, n_steps=1))




groups = [(0, 2), (1, 1)]
hmc_integrators.guess_force_groups(system, nonbonded=0, fft=1, others=0)
integrator = hmc_integrators.GHMCRESPA(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds, groups=groups)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

data["ghmcrespa1"] = pd.DataFrame(sample(context, n_iter=n_iter, n_steps=1))


groups = [(0, 2), (1, 1)]
hmc_integrators.guess_force_groups(system, nonbonded=0, fft=1, others=0)
integrator = hmc_integrators.XHMCRESPAIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds, extra_chances=5, groups=groups)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

data["xhmcrespa1"] = pd.DataFrame(sample(context, n_iter=n_iter, n_steps=1))


data = pd.Panel(data)
print(testsystem)
print(data.mean(1))
print(data.std(1))

#data.iloc[:, :, 0].plot()
