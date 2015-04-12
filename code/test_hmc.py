import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

opencl = mm.Platform_getPlatformByName("OpenCL")
cuda = mm.Platform_getPlatformByName("CUDA")
#platform = opencl
platform = cuda

n_steps = 5000
temperature = 300. * u.kelvin

#testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers)  # Around 1060 molecules of water
testsystem = testsystems.FlexibleWaterBox(box_edge=3.18 * u.nanometers)  # Around 1060 molecules of water
#testsystem = testsystems.AlanineDipeptideExplicit()
system = testsystem.system

integrator = mm.LangevinIntegrator(temperature, 0.25 / u.picoseconds, 0.25 * u.femtoseconds)
context = mm.Context(testsystem.system, integrator, platform)
context.setPositions(testsystem.positions)
context.setVelocitiesToTemperature(temperature)
integrator.step(5000)
positions = context.getState(getPositions=True).getPositions()

def test_hmc(timestep, steps_per_hmc):
    timestep = timestep * u.femtoseconds
    integrator = integrators.HMCIntegrator(temperature, steps_per_hmc, timestep)
    context = mm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(n_steps)
    return integrator.acceptance_rate

#timestep_list = np.linspace(1.5, 2.25, 2)
#timestep_list = [2.25]
timestep_list = [0.4]
#steps_per_hmc_list = np.array([5, 10, 25])
steps_per_hmc_list = np.arange(2, 60)
data = []
for i, timestep in enumerate(timestep_list):
    for j, steps_per_hmc in enumerate(steps_per_hmc_list):
        print(i, j, timestep, steps_per_hmc)
        acceptance = test_hmc(timestep, steps_per_hmc)
        data.append(dict(acceptance=acceptance, timestep=timestep, steps_per_hmc=steps_per_hmc, effective=timestep * acceptance))
        print(steps_per_hmc, acceptance)
        #print(data[-1])
        
data = pd.DataFrame(data)

acceptance = data.pivot("timestep", "steps_per_hmc", "acceptance")
effective = data.pivot("timestep", "steps_per_hmc", "effective")

acceptance.T
#effective
data.to_csv("./data_flexible_water.csv")
