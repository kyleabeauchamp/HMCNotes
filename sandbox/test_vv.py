import lb_loader
import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
pd.set_option('display.width', 1000)

n_steps = 3000
temperature = 300. * u.kelvin
collision_rate = 1.0 / u.picoseconds
timestep = 1.0 * u.femtoseconds
steps_per_hmc = 12

system, positions = lb_loader.load_lb()
positions = lb_loader.pre_equil(system, positions, temperature)



integrator = mm.VerletIntegrator(timestep)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(2)
%timeit integrator.step(200)



integrator = mm.LangevinIntegrator(temperature, 1.0/u.picoseconds, timestep)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(2)
%timeit integrator.step(200)




integrator = hmc_integrators.VelocityVerletIntegrator(timestep)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(2)
%timeit integrator.step(200)





integrator = hmc_integrators.GHMC2(temperature, 100, timestep)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(2)
%timeit integrator.step(2)


integrator = hmc_integrators.GHMC2(temperature, 50, timestep)
integrator.getNumComputations()
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(1)
%timeit integrator.step(4)

for k in range(integrator.getNumComputations()):
    print(integrator.getComputationStep(k))


groups = [(0, 1)]
integrator = hmc_integrators.GHMCRESPA(temperature, 50, timestep, collision_rate, groups)
integrator.getNumComputations()

context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(1)
%timeit integrator.step(4)


for k in range(integrator.getNumComputations()):
    print(integrator.getComputationStep(k))




[1, 'x1', 'x']
[3, '', '']
[1, 'v', 'v+0.5*dt*f/m+(x-x1)/dt']
[4, '', '']
[1, 'v', 'v+0.5*dt*f/m']
[1, 'x', 'x+dt*v']
[1, 'x1', 'x']





[1, 'x', 'x+(dt/1)*v']
[3, '', '']
[1, 'v', '(x-x1)/(dt/1)']
[1, 'v', 'v+0.5*(dt/1)*f0/m']
[4, '', '']
[1, 'v', 'v+0.5*(dt/1)*f0/m']
[1, 'x1', 'x']


        for step in range(self.steps_per_hmc):
            self.addComputePerDof("v", "v+0.5*dt*f/m")
            self.addComputePerDof("x", "x+dt*v")
            self.addComputePerDof("x1", "x")
            self.addConstrainPositions()
            self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
            self.addConstrainVelocities()




        for i in range(stepsPerParentStep):
            self.addComputePerDof("v", "v+0.5*(dt/%s)*f%s/m" % (str_sub, str_group))
            if len(groups) == 1:                
                self.addComputePerDof("x1", "x")
                self.addComputePerDof("x", "x+(dt/%s)*v" % (str_sub))
                self.addConstrainPositions()
                self.addComputePerDof("v", "(x-x1)/(dt/%s)" % (str_sub))
            else:
                self._create_substeps(substeps, groups[1:])
            self.addComputePerDof("v", "v+0.5*(dt/%s)*f%s/m" % (str_sub, str_group))
            
