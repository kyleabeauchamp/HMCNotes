import statsmodels.api as sm
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

#filename0 = "./data/mixed_alanine_LangevinIntegrator_2.000_0.%s"
#filename0 = "./data/mixed_alanine_XCHMCIntegrator_4.598_1.%s"
#filename1 = "./data/mixed_alanine_XCHMCIntegrator_4.029_1.%s"
#filename1 = "./data/mixed_alanine_XCGHMCIntegrator_3.988_100.%s"
#filename1 = "./data/mixed_alanine_LangevinIntegrator_2.000_100.%s"
#filename1 = "./data/mixed_alanine_XCHMCIntegrator_4.000_1.%s"

filename0 = "./data/mixed_alanineexplicit_LangevinIntegrator_2.000_0.%s"
#filename1 = "./data/mixed_alanineexplicit_XCHMCIntegrator_2.883_1.%s"
filename1 = "./data/mixed_alanineexplicit_XCHMCRESPAIntegrator_2.449_1.%s"

top = md.load(filename0 % "pdb")

x0 = pd.read_csv(filename0 % "csv")
x1 = pd.read_csv(filename1 % "csv")

e0 = x0["Potential Energy (kJ/mole)"]
e1 = x1["Potential Energy (kJ/mole)"]

#plot(x0["Elapsed Time (s)"], x0["Potential Energy (kJ/mole)"])
#plot(x1["Elapsed Time (s)"], x1["Potential Energy (kJ/mole)"])

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

trajectories0 = [md.load(filename0 % "dcd", top=top)]
trajectories1 = [md.load(filename1 % "dcd", top=top)]

X0 = msmbuilder.featurizer.DihedralFeaturizer().fit_transform(trajectories0)
X1 = msmbuilder.featurizer.DihedralFeaturizer().fit_transform(trajectories1)

c0 = X0[0][:, 3]
c1 = X1[0][:, 3]

phi0, psi0 = msmbuilder.featurizer.DihedralFeaturizer(sincos=False).fit_transform(trajectories0)[0].T * 180 / pi
phi1, psi1 = msmbuilder.featurizer.DihedralFeaturizer(sincos=False).fit_transform(trajectories1)[0].T * 180 / pi
ass0 = schwalbe_couplings.assign(phi0, psi0)
ass1 = schwalbe_couplings.assign(phi1, psi1)

ind0 = (ass0 == 1).astype('float')  # This state has highest population
ind1 = (ass1 == 1).astype('float')  # This state has highest population

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

print(r0, r1, r1 / r0)

figure()
hexbin(phi0, psi0)
figure()
hexbin(phi1, psi1)


figure()
nlags = 100

dt0 = T0 / len(e0)
dt1 = T1 / len(e1)

acf0 = sm.tsa.acf(c0, nlags=nlags)
acf1 = sm.tsa.acf(c1, nlags=nlags)
plot(dt0 * arange(nlags + 1), acf0, label=filename0[7:-3])
plot(dt1 * arange(nlags + 1), acf1, label=filename1[7:-3])
legend()

pop0 = ind0.cumsum() / np.arange(1, len(ind0) + 1)
pop1 = ind1.cumsum() / np.arange(1, len(ind1) + 1)
pop1 = pop1[0:len(t1)]

figure()
plot(t0, pop0)
plot(t1, pop1)
