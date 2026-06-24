import subprocess

scripts = ["jsm_SubEvo_DF_5.py", "jsm_SubEvo_DF_4.py"]

for script in scripts:

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    subprocess.run(["python", script])
