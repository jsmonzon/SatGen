import subprocess

# print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# print("running first chain")
# print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# subprocess.run("python 2p.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running second chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python 6p_allwalk.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running third chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python 6p_Ncorr_allwalk.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running fourth chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python 6p_fixed.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running fifth chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python 6p_Ncorr_fixed.py", shell=True)