import scipy.integrate
import simtk.openmm as mm
import pandas as pd

box = 4.0
cutoff = box / 2.
q = 0.1
epsilon = 0.0

system = mm.System()
system.setDefaultPeriodicBoxVectors((box, 0, 0), (0, box, 0), (0, 0, box))
system.addParticle(1.0)
system.addParticle(1.0)

f = mm.NonbondedForce()
f.setCutoffDistance(cutoff)
f.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)

f.setUseDispersionCorrection(False)
f.setUseSwitchingFunction(False)
f.setSwitchingDistance(0.9 * cutoff)

f.addParticle(q, 1.0, epsilon)
f.addParticle(-q, 1.0, epsilon)
system.addForce(f)

n = 1000
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
context.setVelocities([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
context.setPositions([(0.0, 0.0, 0.0), (0., 1.0, 0.0)])

data = []
for i in range(n):
    integrator.step(1)
    xcur = integrator.getPerDofVariableByName("xcur")
    fsum = integrator.getPerDofVariableByName("fsum")
    fcur = integrator.getPerDofVariableByName("fcur")
    steps = integrator.getGlobalVariableByName("steps")
    ecur = integrator.getGlobalVariableByName("ecur")
    di = dict(steps=steps, xcur=xcur[1][0], fsum=fsum[1][0], fcur=fcur[1][0], ecur=ecur)
    print(di)
    data.append(di)

data = pd.DataFrame(data)

plot(data.xcur, data.ecur, label="OMM energy")
xlabel("position")
ylabel("energy")

figure()
integrated = scipy.integrate.cumtrapz(-1 * data.fcur.values)
#integrated /= integrated.min()
#integrated *= data.ecur.min()
plot(data.xcur[:-1], integrated, 'g', label="integrated force")
legend(loc=0)




figure()
plot(data.xcur, data.fcur)
xlabel("position")
ylabel("force (1 component)")


