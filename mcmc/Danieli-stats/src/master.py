import subprocess
import json

# Define your global variables here
config = {
    "location": "server",
    "a_stretch": 2.0,
    "nwalk": 100,
    "nstep": 1500,
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

mock_scripts = ["model_2.py", "model_3.py", "model_4.py"]

for idx, mock_script in enumerate(mock_scripts, start=1):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"running {idx} mock")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    subprocess.run(["python", mock_script])
