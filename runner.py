import os
import sys

dry_run = '--dry-run' in sys.argv
detach  = '--detach' in sys.argv
gpu     = '--gpu' in sys.argv

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")


networks_prefix = "networks"

base_networks = {
    }


free_gpus = [7]
# free_gpus = range(8)

jobs = []

learning_rate_options = [1e-4]
spacing_options = [
        (10,0,10),
        # (10,5,10),
        # (10,10,10),
    ]

for learning_rate in learning_rate_options:
    for spacing in spacing_options:
        job = {
                "n_context": spacing[0],
                "n_skip": spacing[1],
                "n_predict": spacing[2],
                "learning_rate": learning_rate,

                "dim_hidden": 1000,
                "gpu": gpu,
            }
        jobs.append(job)

if gpu:
    if len(jobs) > len(free_gpus):
        raise Exception("More GPU jobs specified than GPUs.")
    for i, job in enumerate(jobs):
        job['_gpuid'] = free_gpus[i]

if dry_run:
    print "NOT starting jobs:"
else:
    print "Starting jobs:"

for job in jobs:
    jobname = "skip"
    flagstring = ""
    for flag in job:
        if flag[0] == '_':
            continue
        if isinstance(job[flag], bool):
            if job[flag]:
                jobname = jobname + "_" + flag
                flagstring = flagstring + " --" + flag
            else:
                print "WARNING: Excluding 'False' flag " + flag
        elif flag == 'import':
            imported_network_name = job[flag]
            if imported_network_name in base_networks.keys():
                network_location = base_networks[imported_network_name]
                jobname = jobname + "_" + flag + "_" + str(imported_network_name)
                flagstring = flagstring + " --" + flag + " " + str(network_location)
            else:
                jobname = jobname + "_" + flag + "_" + str(job[flag])
                flagstring = flagstring + " --" + flag + " " + networks_prefix + "/" + str(job[flag])
        else:
            jobname = jobname + "_" + flag + "_" + str(job[flag])
            flagstring = flagstring + " --" + flag + " " + str(job[flag])
    flagstring = flagstring + " --name " + jobname

    jobcommand = "th main.lua" + flagstring
    if gpu:
        jobcommand = "env CUDA_VISIBLE_DEVICES=" + str(job['_gpuid']) + " " + jobcommand

    print(jobcommand)
    if not dry_run:
        if detach:
            os.system(jobcommand + ' 2> logs/' + jobname + '.err 1> logs/' + jobname + '.out &')
        else:
            os.system(jobcommand)

