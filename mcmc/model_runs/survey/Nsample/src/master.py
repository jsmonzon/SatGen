import subprocess

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running model 10 host chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python model_10.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running model 50 host chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python model_50.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running model 500 host chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python model_500.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running model 1000 host chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python model_1000.py", shell=True)

