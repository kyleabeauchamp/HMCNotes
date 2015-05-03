import pymbar
import itertools
import pandas as pd
import lb_loader
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems
pd.set_option('display.width', 1000)

collision_rate = 10000.0 / u.picoseconds
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


timestep = 75 * u.femtoseconds  # LJ Cluster
n_iter = 7000000
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
<openmmtools.testsystems.LennardJonesCluster object at 0x7f1f49314550>
             ghmc      ghmc1  ghmcrespa1   langevin          xhmc1  xhmcrespa1
energy  29.958707  29.965658   29.940944  29.948993  374714.786309   29.942151
            ghmc     ghmc1  ghmcrespa1  langevin         xhmc1  xhmcrespa1
energy  8.624575  8.657256    8.641558  8.630257  6.614858e+08    8.632509
                     g             mu         sigma         stderr      zerr        relerr
ghmc        145.388573      29.958707  8.624575e+00       0.039306  0.247135      0.000324
ghmc1        30.496604      29.965658  8.657256e+00       0.018070  0.922208      0.000556
ghmcrespa1   23.654946      29.940944  8.641558e+00       0.015886 -0.506711     -0.000269
langevin     27.029683      29.948993  8.630257e+00       0.016959  0.000000      0.000000
xhmc1         2.244405  374714.786309  6.614858e+08  374560.642342  1.000332  12510.765617
xhmcrespa1   13.940385      29.942151  8.632509e+00       0.012182 -0.561653     -0.000228

"""
