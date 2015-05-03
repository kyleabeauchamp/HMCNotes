import pymbar
import itertools
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems
pd.set_option('display.width', 1000)

collision_rate = 10000.0 / u.picoseconds
temperature = 25. * u.kelvin

testsystem = testsystems.LennardJonesFluid()

system, positions = testsystem.system, testsystem.positions

positions = np.loadtxt("./sandbox/ljbox.dat")
length = 2.66723326712
boxes = ((length, 0, 0), (0, length, 0), (0, 0, length))
system.setDefaultPeriodicBoxVectors(*boxes)

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.5 * u.femtoseconds)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)
integrator.step(20000)
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


timestep = 2 * u.femtoseconds  # LJ Cluster
n_iter = 160000
data = {}

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, timestep / 2.)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

data["langevin"] = pd.DataFrame(sample(context, n_iter=n_iter, n_steps=40))


integrator = integrators.GHMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=collision_rate)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)


data["ghmc"] = pd.DataFrame(sample(context, n_iter=n_iter, n_steps=1))


integrator = integrators.GHMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)


data["ghmc1"] = pd.DataFrame(sample(context, n_iter=n_iter, n_steps=1))


integrator = integrators.XHMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds, extra_chances=5)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

data["xhmc1"] = pd.DataFrame(sample(context, n_iter=n_iter, n_steps=1))




groups = [(0, 2), (1, 1)]
integrators.guess_force_groups(system, nonbonded=0, fft=1, others=0)
integrator = integrators.GHMCRESPA(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds, groups=groups)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

data["ghmcrespa1"] = pd.DataFrame(sample(context, n_iter=n_iter, n_steps=1))


groups = [(0, 2), (1, 1)]
integrators.guess_force_groups(system, nonbonded=0, fft=1, others=0)
integrator = integrators.XHMCRESPAIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds, extra_chances=5, groups=groups)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

data["xhmcrespa1"] = pd.DataFrame(sample(context, n_iter=n_iter, n_steps=1))


data = pd.Panel(data)
print(testsystem)
print(data.mean(1))
print(data.std(1))

#data.iloc[:, :, 0].plot()

X = dict()
X["mu"] = [data[key].iloc[:, 0].mean() for key in data.keys()]
X["sigma"] = [data[key].iloc[:, 0].std() for key in data.keys()]
X["g"] = [pymbar.timeseries.statisticalInefficiency(data[key].iloc[:, 0]) for key in data.keys()]

X = pd.DataFrame(X, index=data.keys())
X['stderr'] = X.sigma * (n_iter / X.g) ** -0.5
X["zerr"] = (X.mu - X.mu.langevin) / X.stderr
X["relerr"] = (X.mu - X.mu.langevin) / X.mu.langevin

print(X)
print((X.mu - X.mu.langevin) / X.stderr)

"""
                      g           mu     sigma    stderr       zerr    relerr
ghmc         157.665521 -3997.648116  5.643862  0.177168 -11.109699  0.000493
ghmc1       1385.123790 -3997.317036  5.609147  0.521893  -3.137047  0.000410
ghmcrespa1  1192.634591 -3997.397102  5.287009  0.456461  -3.762132  0.000430
langevin      23.515424 -3995.679835  5.599026  0.067878   0.000000 -0.000000
xhmc1       1573.943076 -3996.371055  5.805638  0.575817  -1.200416  0.000173
xhmcrespa1  1559.874999 -3998.137542  5.659271  0.558786  -4.398299  0.000615
"""
