import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

collision_rate = 1.0 / u.picoseconds
n_steps = 1500
temperature = 300. * u.kelvin

testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers)  # Around 1060 molecules of water
system = testsystem.system

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.25 * u.femtoseconds)
context = mm.Context(testsystem.system, integrator)
context.setPositions(testsystem.positions)
context.setVelocitiesToTemperature(temperature)
integrator.step(5000)
positions = context.getState(getPositions=True).getPositions()

def test_hmc(timestep, steps_per_hmc):
    timestep = timestep * u.femtoseconds
    integrator = integrators.GHMC2(temperature, steps_per_hmc, timestep)
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(n_steps)
    return integrator.acceptance_rate

timestep_list = np.linspace(1.5, 2.25, 3)
steps_per_hmc_list = np.array([5, 10, 25])
#steps_per_hmc_list = np.array([10, 5, 25])
data = []
for i, timestep in enumerate(timestep_list):
    for j, steps_per_hmc in enumerate(steps_per_hmc_list):
        print(i, j, timestep, steps_per_hmc)
        acceptance = test_hmc(timestep, steps_per_hmc)
        data.append(dict(acceptance=acceptance, timestep=timestep, steps_per_hmc=steps_per_hmc, normalized=timestep * acceptance))
        print(data[-1])
        
data = pd.DataFrame(data)

acceptance = data.pivot("timestep", "steps_per_hmc", "acceptance")
normalized = data.pivot("timestep", "steps_per_hmc", "normalized")

acceptance
normalized

"""
In [25]: acceptance
Out[25]: 
steps_per_hmc        5      10        25
timestep                                
1.500          0.791333  0.760  0.664000
1.875          0.694000  0.724  0.666667
2.250          0.601333  0.680  0.634000

In [26]: normalized
Out[26]: 
steps_per_hmc       5       10      25
timestep                              
1.500          1.18700  1.1400  0.9960
1.875          1.30125  1.3575  1.2500
2.250          1.35300  1.5300  1.4265
"""
