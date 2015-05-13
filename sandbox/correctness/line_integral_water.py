from simtk import unit as u
import openmmtools
import scipy.integrate
import simtk.openmm as mm
import pandas as pd

testsystem = openmmtools.testsystems.WaterBox(box_edge=0.5*u.nanometers)
system, positions = testsystem.system, testsystem.positions

z_offset = 0.75
box = 3.0
cutoff = box / 2.
r = 0.09572
theta = 1.82421813418
c = np.cos(theta / 2.)
s = np.sin(theta / 2.)

positions = [(0.0, 0.0, 0.0), (r * c, r * s, 0.0), (r * c, -r * s, 0.0)]
positions.extend([(0.0, 0.0, z_offset), (r * c, r * s, z_offset), (r * c, -r * s, z_offset)])

system.setDefaultPeriodicBoxVectors((box, 0, 0), (0, box, 0), (0, 0, box))

f = system.getForce(2)
f.setCutoffDistance(cutoff)
f.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)

f.setUseDispersionCorrection(False)
f.setUseSwitchingFunction(True)
f.setSwitchingDistance(cutoff - 0.5)

n = 5000
dt = (box) / n
integrator = mm.CustomIntegrator(dt)

integrator.addPerDofVariable("xcur", 0.0)
integrator.addPerDofVariable("fcur", 0.0)
integrator.addPerDofVariable("fsum", 0.0)
integrator.addGlobalVariable("steps", 0.0)
integrator.addGlobalVariable("ecur", 0.0)


integrator.addComputePerDof("x", "x + dt * v")
integrator.addComputePerDof("xcur", "x")
integrator.addComputePerDof("fsum", "fsum + f")
integrator.addComputePerDof("fcur", "f")
integrator.addComputeGlobal("steps", "steps + 1")
integrator.addComputeGlobal("ecur", "energy")

context = mm.Context(system, integrator)
velocities = [(0.0, 0.0, 0.0)] * 3
velocities.extend([(1.0, 0.0, 0.0)] * 3)
context.setVelocities(velocities)
context.setPositions(positions)

data = []
for i in range(n):
    integrator.step(1)
    xcur = integrator.getPerDofVariableByName("xcur")
    fsum = integrator.getPerDofVariableByName("fsum")
    fcur = integrator.getPerDofVariableByName("fcur")
    steps = integrator.getGlobalVariableByName("steps")
    ecur = integrator.getGlobalVariableByName("ecur")
    di = dict(steps=steps, xcur=xcur[3][0], fsum=fsum[3][0], fcur=fcur[3][0], ecur=ecur)
    print(di)
    data.append(di)

data = pd.DataFrame(data)

integrated = scipy.integrate.cumtrapz(-1 * data.fcur.values)
plot(data.xcur[:-1], integrated, 'g', label="integrated force")
xlabel("position")
ylabel("energy")
title("Raw integrated force")


figure()
plot(data.xcur, data.ecur, label="OMM energy")
xlabel("position")
ylabel("energy")


integrated /= integrated.max()
integrated *= data.ecur.max()
plot(data.xcur[:-1], integrated, 'g', label="integrated force")
legend(loc=0)



figure()
plot(data.xcur, data.fcur)
xlabel("position")
ylabel("force (1 component)")


