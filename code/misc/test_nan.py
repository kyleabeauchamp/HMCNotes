from simtk import unit as u
import simtk.openmm as mm
import openmmtools

testsystem = openmmtools.testsystems.WaterBox()
system, positions = testsystem.system, testsystem.positions

integrator = mm.CustomIntegrator(1.75 * u.femtoseconds)

temperature = 300 * u.kelvin

integrator.addPerDofVariable("x1", 0)
integrator.addPerDofVariable("x2", 0)
integrator.addPerDofVariable("x3", 0)
integrator.addGlobalVariable("e1", 0)
integrator.addGlobalVariable("e2", 0)
integrator.addGlobalVariable("e3", 0)
integrator.addGlobalVariable("e4", 0)
integrator.addGlobalVariable("e5", 0)
integrator.addGlobalVariable("one", 1.0)
integrator.addGlobalVariable("e2m", 0.0)

integrator.addUpdateContextState()
integrator.addComputePerDof("v", "v+0.5*dt*f/m")
integrator.addComputePerDof("x", "x+dt*v")
integrator.addComputePerDof("x1", "x")
integrator.addConstrainPositions()
integrator.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
integrator.addConstrainVelocities()

integrator.addComputePerDof("x2", "x")
integrator.addComputePerDof("x3", "abs(step(x) - step(- x))")

integrator.addComputeGlobal("e1", "energy")
integrator.addComputeGlobal("e2", "abs(step(e1) - step(- e1))")
integrator.addComputeGlobal("e2m", "one - e2")
integrator.addComputeGlobal("e3", "e1 * e2")
#integrator.addComputeGlobal("e4", "e1 * (one - e2m)")
integrator.addComputeGlobal("e4", "step(gaussian)")
integrator.addComputeGlobal("e5", "select(e4, 1.0, 2.0)")



context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

for i in range(5):
    integrator.step(1)
    energy = context.getState(getEnergy=True).getPotentialEnergy()
    x2 = integrator.getPerDofVariableByName("x2")[0:3]
    x3 = integrator.getPerDofVariableByName("x3")[0:3]
    e1 = integrator.getGlobalVariableByName("e1")
    e2 = integrator.getGlobalVariableByName("e2")
    e3 = integrator.getGlobalVariableByName("e3")
    e4 = integrator.getGlobalVariableByName("e4")
    e5 = integrator.getGlobalVariableByName("e5")
    print(x2)
    print(x3)
    print(e4, e5)


