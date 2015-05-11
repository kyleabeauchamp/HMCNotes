import copy
import scipy.misc
import scipy.integrate
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

sysname = "ljbox"

system, positions, groups, temperature, timestep = lb_loader.load(sysname)

integrator = hmc_integrators.GHMCIntegratorOneStep(temperature, timestep=8*u.femtoseconds)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(10)
positions0 = context.getState(getPositions=True).getPositions(asNumpy=True)

i, j = 0, 0

def obj(x):
    positions = copy.deepcopy(positions0)
    positions[i][j] = positions[i][j] + x * u.nanometers
    context.setPositions(positions)
    state = context.getState(getPositions=True, getForces=True, getEnergy=True)
    f = state.getForces()[i][j]
    e = state.getPotentialEnergy() / u.kilojoules_per_mole
    return e, f

f = lambda x: -obj(x)[0]

fhat = scipy.misc.derivative(f, 0.0, dx=2E-5)
f0 = obj(0.0)[1]
f0 = f0 / f0.unit
fhat, f0, fhat - f0, (fhat - f0) / f0

intf = lambda x: obj(x)[1] / (u.kilojoule_per_mole / u.nanometer)
x0, x1 = -0.1, 0.1

ehat, ehaterr = scipy.integrate.quad(intf, x0, x1)
e0 = f(x1) - f(x0)
ehat, e0, ehat - e0, (ehat - e0) / e0
