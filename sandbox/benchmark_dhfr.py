import numpy as np
import pandas as pd
import time
import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

n_steps = 10000
temperature = 300. * u.kelvin

data = {}
for cutoff in np.linspace(0.8, 1.3, 10):

    testsystem = testsystems.DHFRExplicit(nonbondedCutoff=cutoff * u.nanometers, nonbondedMethod=app.PME)
    system, positions = testsystem.system, testsystem.positions

    integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 2.0 * u.femtoseconds)
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)

    integrator.step(1)
    t0 = time.time()
    integrator.step(n_steps)
    dt = time.time() - t0

    time_per_frame = dt / float(n_steps) * u.seconds

    frames_per_day = time_per_frame / u.day

    ns_per_day = (2 * u.femtoseconds / frames_per_day) / u.nanoseconds
    data[cutoff] = ns_per_day
    print(cutoff, ns_per_day)
