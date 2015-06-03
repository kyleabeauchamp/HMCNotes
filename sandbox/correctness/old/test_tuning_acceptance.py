import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

precision = "mixed"

sysname = "switchedaccuratenptwater"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(testsystem, temperature, timestep, steps=equil_steps, minimize=True, steps_per_hmc=steps_per_hmc)


timestep = 5.65 * u.femtoseconds
steps_per_hmc = 250
extra_chances = 6
groups = [(0, 2), (1, 1)]
integrator = hmc_integrators.UnrolledXCHMCRESPAIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, groups=groups)


"""
timestep = 3.99 * u.femtoseconds
extra_chances = 8
steps_per_hmc = 105
integrator = hmc_integrators.UnrolledXCHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances)
"""

simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision)
integrator.step(600)

out = integrator.vstep(10)
integrator.all_counts
integrator.acceptance_rate
integrator.steps_accepted, integrator.steps_taken

print(pd.DataFrame([integrator.summary()]).to_string(formatters=[lambda x: "%.4g" % x for x in range(25)]))
c = integrator.all_counts
p = integrator.all_probs
c
p
integrator.effective_timestep, integrator.effective_ns_per_day
