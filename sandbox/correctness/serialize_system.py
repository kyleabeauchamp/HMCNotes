import lb_loader
import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
import sys

def write_file(filename, contents):
    with open(filename, 'w') as outfile:
        outfile.write(contents)

def run(sysname):
    system_filename = "./systems/%s_system.xml" % sysname
    state_filename = "./systems/%s_state.xml" % sysname

    system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
    positions, boxes, state = lb_loader.equilibrate(testsystem, temperature, langevin_timestep, steps=equil_steps, minimize=True)

    write_file(system_filename, mm.XmlSerializer.serialize(system))
    serialized = mm.XmlSerializer.serialize(state)
    write_file(state_filename, serialized)


if __name__ == "__main__":
    sysname  = sys.argv[1]
    run(sysname)
