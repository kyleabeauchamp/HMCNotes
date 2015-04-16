import copy
import time
import lb_loader
import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems
pd.set_option('display.width', 1000)

n_steps = 1000
temperature = 300. * u.kelvin
collision_rate = 1.0 / u.picoseconds
timestep = 1.0 * u.femtoseconds

testsystem = testsystems.DHFRExplicit()
system, positions = testsystem.system, testsystem.positions
integrators.guess_force_groups(system, nonbonded=1, fft=1, others=0)
positions = lb_loader.pre_equil(system, positions, temperature)
groups = [(0, 2), (1, 1)]

idict = {
"verlet": mm.VerletIntegrator(timestep), 
"langevin": mm.LangevinIntegrator(temperature, collision_rate, timestep),
"vv": integrators.VelocityVerletIntegrator(timestep),
"vvvr": integrators.VelocityVerletIntegrator(timestep),
"ghmc1": integrators.GHMCIntegratorOneStep(temperature=temperature, collision_rate=collision_rate, timestep=timestep),
"ghmc10": integrators.GHMCIntegrator(temperature=temperature, collision_rate=collision_rate, timestep=timestep, steps_per_hmc=10),
"ghmc20": integrators.GHMCIntegrator(temperature=temperature, collision_rate=collision_rate, timestep=timestep, steps_per_hmc=20),
"ghmcrespa20": integrators.GHMCRESPA(temperature, steps_per_hmc=20, timestep=timestep, collision_rate=collision_rate, groups=groups)
}

factors = {"ghmc10":10, "ghmc20":20, "ghmcrespa20":20}


data = []
for name, integrator in idict.items():
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(1)    
    t0 = time.time()
    cur_steps = n_steps / factors.get(name, 1)
    integrator.step(cur_steps)
    dt = time.time() - t0
    ns_per_day = 0.002 / dt * 24 * 60 * 60
    data.append(dict(name=name, dt=dt, ns_per_day=ns_per_day))
    print(data[-1])

data = pd.DataFrame(data)
data.to_csv("./tables/raw_performance.csv")
