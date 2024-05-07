import subprocess
import json

mock_dirs = ["../s15_data/mock_1_0/", "../s15_data/mock_1_1/", "../s15_data/mock_1_2/"]
             #"../s30_data/mock_2_0/", "../s30_data/mock_2_1/", "../s30_data/mock_2_2/"]

config = {
    "location": "local",
    "a_stretch": 2.0,
    "nwalk": 15,
    "nstep": 30,
    "ncores": 8,
    "min_mass": 6.5,
    "max_mass":11.5,
    "Nsamp": 1,
    "N_bin": 31,
    "init_gauss": 1e-2,
    "N_corr": True,
    "p0_corr": True,
    "savefig": True,
    "reset": True,
    }

for idx, dir in enumerate(mock_dirs, start=0):

    config["savedir"] = dir 

    # Write the configuration to a JSON file
    with open("config.json", "w") as f:
        json.dump(config, f)

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"running chain in {dir}")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    subprocess.run(["python", "chain_runner.py"])
