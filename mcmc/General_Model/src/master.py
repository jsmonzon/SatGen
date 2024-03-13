import subprocess
import json

# Define your global variables here
config = {
    "location": "server",
    "a_stretch": 2.0,
    "nwalk": 100,
    "nstep": 100,
    "ncores": 16,
    "min_mass": 6.5,
    "Nsamp": 1,
    "init_gauss": 1e-3,
    "N_corr": True,
    "p0_corr": True,
    "savefig": True,
    "reset": True,
}

# Write the configuration to a JSON file
with open("config.json", "w") as f:
    json.dump(config, f)

mock_scripts = ["mock_L1.py", "mock_L2.py", "mock_min_mass.py", "mock_N_host.py"]

for idx, mock_script in enumerate(mock_scripts, start=1):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"running {idx} mock")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    subprocess.run(["python", mock_script])
