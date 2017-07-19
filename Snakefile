from experiments import enumerate_experiments, EXPERIMENTS

enumerate_experiments()
print(EXPERIMENTS)

DCD_FILENAMES = ["data/{sysname}/{intname}.dcd".format(sysname=sysname, intname=intname) for (sysname, intname) in EXPERIMENTS.keys()]
SYSNAMES = list({sysname for (sysname, intname) in EXPERIMENTS.keys()})
INTNAMES = list({intname for (sysname, intname) in EXPERIMENTS.keys()})
SYSNAME_CONSTRAINT = "(%s)" % "|".join(SYSNAMES)
INTNAME_CONSTRAINT = "(%s)" % "|".join(INTNAMES)

#N_EFF = 100000
N_EFF = 10000

wildcard_constraints:
    sysname=SYSNAME_CONSTRAINT,
    integrator=INTNAME_CONSTRAINT


rule all:
    input:
        dcd=DCD_FILENAMES,
        csv = expand("data/{sysname}_summary.csv", sysname=SYSNAMES),

rule serialize_integrators:
    output:
        integrator="data/integrators/{sysname}/{integrator}.pkl",
    shell:
        "serialize_integrators.py {wildcards.sysname} {wildcards.integrator} {output.integrator}"

rule serialize_systems:
    output:
        system="data/systems/{sysname}_state.xml",
        state="data/systems/{sysname}_system.pkl",
    shell:
        "serialize_system.py {wildcards.sysname} {output.system} {output.state};"

rule run_production:
    input:
        system="data/systems/{sysname}_state.xml",
        state="data/systems/{sysname}_system.pkl",
        integrator="data/integrators/{sysname}/{integrator}.pkl",
    output:
        dcd="data/{sysname}/{integrator}.dcd",
        csv="data/{sysname}/{integrator}.csv",
    shell:
        "converge.py {input.system} {input.state} {input.integrator} {wildcards.sysname} {N_EFF} {output.csv} {output.dcd}"


rule summarize_one:
    input:
        csv="data/{sysname}/{integrator}.csv",
    output:
        csv="data/summary/{sysname}_{integrator}.csv",
    shell:
        "summarize.py one {input.csv} {output.csv};"


rule summarize_all:
    input:
        csvs = lambda wildcards: ["data/summary/{sysname}_{integrator}.csv".format(integrator=integrator, sysname=wildcards.sysname) for (sysname, integrator) in EXPERIMENTS.keys() if sysname == wildcards.sysname]
    output:
        csv = "data/{sysname}_summary.csv",
    params:
        csvs = lambda wildcards: ",".join(["data/summary/{sysname}_{integrator}.csv".format(integrator=integrator, sysname=wildcards.sysname) for (sysname, integrator) in EXPERIMENTS.keys() if sysname == wildcards.sysname])
    shell:
        "summarize.py many {params.csvs} {output.csv};"
