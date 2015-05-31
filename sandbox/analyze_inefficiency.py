import mdtraj as md
import msmbuilder.decomposition, msmbuilder.featurizer
import pymbar
import pandas as pd

#filename0 = "./data/mixed_alanineexplicit_LangevinIntegrator_2.000_0.%s"
filename0 = "./data/mixed_alanineexplicit_XCHMCRESPAIntegrator_2.000_0.%s"
#filename1 = "./data/mixed_alanineexplicit_XCHMCRESPAIntegrator_2.886_0.%s"
filename1 = "./data/mixed_alanineexplicit_XCHMCRESPAIntegrator_4.000_0.%s"

x0 = pd.read_csv(filename0 % "csv")[10:]
x1 = pd.read_csv(filename1 % "csv")[10:]

plot(x0["Elapsed Time (s)"], x0["Potential Energy (kJ/mole)"])
plot(x1["Elapsed Time (s)"], x1["Potential Energy (kJ/mole)"])

t0 = x0["Elapsed Time (s)"]
t1 = x1["Elapsed Time (s)"]

T0 = x0["Elapsed Time (s)"].max()
T1 = x1["Elapsed Time (s)"].max()


g0 = pymbar.timeseries.statisticalInefficiency(x0["Potential Energy (kJ/mole)"])
g1 = pymbar.timeseries.statisticalInefficiency(x1["Potential Energy (kJ/mole)"])

n0 = len(x0) / g0
n1 = len(x1) / g1

r0 = n0 / T0
r1 = n1 / T1

print(g0, g1)
print(r1 / r0)

#####

trajectories0 = [md.load(filename0 % "dcd", top=(filename0 % "pdb"))]
trajectories1 = [md.load(filename1 % "dcd", top=(filename1 % "pdb"))]

X0 = msmbuilder.featurizer.DihedralFeaturizer().fit_transform(trajectories0)
X1 = msmbuilder.featurizer.DihedralFeaturizer().fit_transform(trajectories1)

tica0 = msmbuilder.decomposition.tica.tICA().fit(X0)
tica1 = msmbuilder.decomposition.tica.tICA().fit(X1)

y0 = X0[0][:, 0]
y1 = X1[0][:, 0]
#y0 = tica0.transform(X0)[0][:, 0]
#y1 = tica1.transform(X1)[0][:, 0]

g0 = pymbar.timeseries.statisticalInefficiency(y0)
g1 = pymbar.timeseries.statisticalInefficiency(y1)

n0 = len(y0) / g0
n1 = len(y1) / g1

r0 = n0 / T0
r1 = n1 / T1

r1 / r0


#####

phi0, psi0 = msmbuilder.featurizer.DihedralFeaturizer(sincos=False).fit_transform(trajectories0)[0].T * 180 / pi
phi1, psi1 = msmbuilder.featurizer.DihedralFeaturizer(sincos=False).fit_transform(trajectories1)[0].T * 180 / pi

#plot(t0, phi0)
#figure()
#plot(t1, phi1)

plot(phi0, psi0, 'x')
figure()
plot(phi1, psi1, 'x')
