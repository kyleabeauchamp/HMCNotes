import schwalbe_couplings
import mdtraj as md
import msmbuilder.decomposition, msmbuilder.featurizer, msmbuilder.msm
import pymbar
import pandas as pd

lag_time = 25

#filename0 = "./data/mixed_alanineexplicit_LangevinIntegrator_2.000_0.%s"
#filename0 = "./data/mixed_alanineexplicit_XCHMCRESPAIntegrator_2.000_0.%s"
#filename1 = "./data/mixed_alanineexplicit_XCHMCRESPAIntegrator_2.886_0.%s"
#filename1 = "./data/mixed_alanineexplicit_XCHMCRESPAIntegrator_4.000_0.%s"

#filename0 = "./data/mixed_switchedaccuratewater_LangevinIntegrator_2.000_0.%s"
#filename1 = "./data/mixed_switchedaccuratewater_XCHMCRESPAIntegrator_3.631_0.%s"
#filename1 = "./data/mixed_switchedaccuratewater_XCGHMCRESPAIntegrator_3.631_100.%s"
#filename1 = "./data/mixed_switchedaccuratewater_LangevinIntegrator_1.000_0.%s"
#filename1 = "./data/mixed_switchedaccuratewater_LangevinIntegrator_2.000_10.%s"

filename0 = "./data/mixed_alanine_LangevinIntegrator_2.000_100.%s"
#filename1 = "./data/mixed_alanine_LangevinIntegrator_2.000_100.%s"
filename1 = "./data/mixed_alanine_XCHMCIntegrator_4.598_1.%s"

x0 = pd.read_csv(filename0 % "csv")
x1 = pd.read_csv(filename1 % "csv")

e0 = x0["Potential Energy (kJ/mole)"]
e1 = x1["Potential Energy (kJ/mole)"]

plot(x0["Elapsed Time (s)"], x0["Potential Energy (kJ/mole)"])
plot(x1["Elapsed Time (s)"], x1["Potential Energy (kJ/mole)"])

t0 = x0["Elapsed Time (s)"]
t1 = x1["Elapsed Time (s)"]

T0 = x0["Elapsed Time (s)"].max()
T1 = x1["Elapsed Time (s)"].max()

samples_per_sec0 = len(e0) / T0
samples_per_sec1 = len(e1) / T1


g0 = pymbar.timeseries.statisticalInefficiency(e0)
g1 = pymbar.timeseries.statisticalInefficiency(e1)

n0 = len(x0) / g0
n1 = len(x1) / g1

r0 = n0 / T0
r1 = n1 / T1

print(g0, g1, r0, r1, r1 / r0)
print(e0.mean(), e1.mean(), e0.std() * n0 ** -0.5, e1.std() * n1 ** -0.5)

#####

trajectories0 = [md.load(filename0 % "dcd", top=(filename0 % "pdb"))]
trajectories1 = [md.load(filename1 % "dcd", top=(filename0 % "pdb"))]

X0 = msmbuilder.featurizer.DihedralFeaturizer().fit_transform(trajectories0)
X1 = msmbuilder.featurizer.DihedralFeaturizer().fit_transform(trajectories1)

tica0 = msmbuilder.decomposition.tica.tICA(lag_time=lag_time).fit(X0)
tica1 = msmbuilder.decomposition.tica.tICA(lag_time=lag_time).fit(X1)

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
ass0 = schwalbe_couplings.assign(phi0, psi0)
ass1 = schwalbe_couplings.assign(phi1, psi1)

msm0 = msmbuilder.msm.MarkovStateModel(lag_time=lag_time).fit([ass0])
msm1 = msmbuilder.msm.MarkovStateModel(lag_time=lag_time).fit([ass1])

msm0.transmat_.diagonal()
msm1.transmat_.diagonal()

msm0.timescales_
msm1.timescales_

tau0 = msm0.timescales_[0]
tau1 = msm1.timescales_[0]

r0 = samples_per_sec0 / tau0
r1 = samples_per_sec1 / tau1

r0, r1, r1 / r0

figure()
hexbin(phi0, psi0)
figure()
hexbin(phi1, psi1)
