#!/usr/bin/env python
import lb_loader
import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
import sys
import pickle


def run():
    sysname, system_filename, state_filename = sys.argv[1:]

    system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
    positions, boxes, state = lb_loader.equilibrate(testsystem, temperature, langevin_timestep, steps=equil_steps, minimize=True)

    #lb_loader.write_file(system_filename, mm.XmlSerializer.serialize(system))
    pickle.dump(testsystem, open(system_filename, 'wb'))
    serialized = mm.XmlSerializer.serialize(state)
    lb_loader.write_file(state_filename, serialized)


if __name__ == "__main__":
    run()
