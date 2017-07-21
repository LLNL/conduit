import subprocess

def sexe(cmd):
    return subprocess.call(cmd,shell=True)

r = 0
while r ==0:
    r = sexe("ctest -V -R t_relay_mpi_test_crash")
    print r
