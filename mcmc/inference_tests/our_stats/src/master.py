import subprocess
import json

# mock_dirs = ["../s0_data/mock_0_0/", "../s0_data/mock_0_1/", "../s0_data/mock_0_2/",
#              "../s15_data/mock_1_1/", "../s15_data/mock_1_2/",
#              "../s30_data/mock_2_2/"]

mock_dirs = ["../s15_data/mock_1_0/", "../s30_data/mock_2_1/"]
start_vals = [[10.5, 1.93, 0, 0, 0.0, 0], [10.5, 1.81, 0, 0, 0.57, 0]]

config = {
    "location": "server",
    "a_stretch": 1.2,
    "nwalk": 15,
    "nstep": 5000,
    "ncores": 16,
    "min_mass": 6.5,
    "max_mass":11.5,
    "Nsamp": 1,
    "max_N": 1000,
    "init_gauss": 1e-4,
    "N_corr": True,
    "p0_corr": True,
    "savefig": True,
    "reset": True,
    }

for idx, dir in enumerate(mock_dirs, start=0):

    config["savedir"] = dir
    config["start_val"] = start_vals[idx]

    # Write the configuration to a JSON file
    with open("config.json", "w") as f:
        json.dump(config, f)

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"running chain in {dir}")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    subprocess.run(["python", "chain_runner.py"])
