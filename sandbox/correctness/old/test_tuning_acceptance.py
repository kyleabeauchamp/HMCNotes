import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

precision = "mixed"

sysname = "chargedswitchedaccurateljbox"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
equil_steps = equil_steps / 10
positions, boxes = lb_loader.equilibrate(testsystem, temperature, timestep, steps=equil_steps, minimize=True, steps_per_hmc=steps_per_hmc)


#collision_rate = 1E-3 / u.picoseconds

#integrator = hmc_integrators.HMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)
#integrator = hmc_integrators.GHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, collision_rate=collision_rate)
#integrator = hmc_integrators.XCHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)
integrator = hmc_integrators.UnrolledXCHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)


simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision)

integrator.step(300)
integrator.acceptance_rate

steps_per_hmc = 25
integrator = hmc_integrators.XCHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=1)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(10000)
print(pd.DataFrame([integrator.summary()]).to_string(formatters=[lambda x: "%.4g" % x for x in range(25)]))
c = integrator.all_counts
p = integrator.all_probs
c
p
1 - p[0], integrator.fraction_force_wasted, integrator.effective_timestep
