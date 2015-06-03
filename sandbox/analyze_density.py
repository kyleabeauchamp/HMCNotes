import mdtraj as md
import pandas as pd
import pymbar

charges0 = np.array([-0.8, 0.4, 0.4])

filename = "data/mixed_switchedaccuratenptwater_LangevinIntegrator_2.000_0.%s"
#filename = "data/mixed_switchedaccuratenptwater_UnrolledXCHMCRESPAIntegrator_5.750_1.%s"
#filename = "data/mixed_switchedaccuratenptwater_UnrolledXCHMCRESPAIntegrator_5.650_1.%s"

top = md.load("./water.pdb")

data = pd.read_csv(filename % "csv")
trj = md.load(filename % "dcd", top=top)

charges = np.tile(charges0, trj.n_residues)
dipoles = md.geometry.thermodynamic_properties.dipole_moments(trj, charges)

#x = dipoles[:, 0]
x = data["Density (g/mL)"]
#x = data['Potential Energy (kJ/mole)']
plot(x)

T = data["Elapsed Time (s)"].iloc[-1]
g = pymbar.timeseries.statisticalInefficiency(x)

neff = len(x) / g

mu = x.mean()
sigma = x.std()
std = x.std() * neff ** -0.5

neff_per_sec = neff / T

mu, sigma, std, g, neff, std / mu, neff_per_sec
