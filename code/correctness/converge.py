import lb_loader
import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
import sys

def run():
    sysname, intname, Neff_cutoff = sys.argv[1:]

    system_filename = "./data/systems/%s_system.xml" % sysname
    state_filename = "./data/systems/%s_state.xml" % sysname

    csv_filename = "./data/{sysname}/{intname}.csv".format(sysname=sysname, intname=intname)
    pdb_filename = "./data/%s.pdb" % fmt_string
    dcd_filename = "./data/%s.dcd" % fmt_string

    state = mm.XmlSerializer.deserialize(open(state_filename).read())
    system = mm.XmlSerializer.deserialize(open(system_filename).read())

    itype = type(integrator).__name__
    print("%s    %s" % (fmt_string, itype))

    simulation = lb_loader.build(testsystem, integrator, temperature, state=state)
    simulation.runForClockTime(0.5 * u.minutes)

    output_frequency = 100 if "Langevin" in itype else 2
    kineticEnergy = True if "MJHMC" in itype else False
    simulation.reporters.append(app.StateDataReporter(csv_filename, output_frequency, step=True, time=True, potentialEnergy=True, kineticEnergy=kineticEnergy, temperature=True, density=True, elapsedTime=True))
    simulation.reporters.append(app.DCDReporter(dcd_filename, output_frequency))
    lb_loader.converge(simulation, csv_filename, Neff_cutoff)

if __name__ == "__main__":

    run()
