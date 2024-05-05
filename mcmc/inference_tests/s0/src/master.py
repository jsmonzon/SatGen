import subprocess
import json

# Define your global variables here
config = {
    "location": "server",
    "a_stretch": 2.0,
    "nwalk": 15,
    "nstep": 5000,
    "ncores": 16,
    "min_mass": 6.5,
    "max_N": 700,
    "Nsamp": 1,
    "N_bin": 31,
    "init_gauss": 1e-2,
    "N_corr": True,
    "p0_corr": True,
    "savefig": True,
    "reset": True,
}

# Write the configuration to a JSON file
with open("config.json", "w") as f:
    json.dump(config, f)

model_scripts = ["mock_0.py", "mock_1.py"]

for idx, model_script in enumerate(model_scripts, start=0):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"running mock {idx}")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    subprocess.run(["python", model_script])
