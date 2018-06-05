import csv
import subprocess
import copy
import numpy as np

f = open("photonPropagate/datafiles/geo-f2k", 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')

# Save original geometry
orig_geo = []
for line in rdr:
    orig_geo.append(line)
f.close()
new_geo = copy.deepcopy(orig_geo)

# Range to change coordinate(In this case, we will change x)
order = 10
dx = np.arange(-5, 5.1, 0.5)
# dx = np.array([5, 10, 15, 20])
for i in range(len(dx)):
    wf = open("photonPropagate/datafiles/geo-f2k", 'w', encoding='utf-8', newline='')
    wr = csv.writer(wf, delimiter='\t')
    new_geo[1][2] = str(float(orig_geo[1][2]) + dx[i])
    for line in new_geo:
        wr.writerow(line)
    wf.close()
    subprocess.call('cd photonPropagate;WFLA=405 ./ppc 1 1 %d' % pow(10,order), shell=True)

# Recover the file using orig_geo
wf = open("photonPropagate/datafiles/geo-f2k", 'w', encoding='utf-8', newline='')
wr = csv.writer(wf, delimiter='\t')
for line in orig_geo:
    wr.writerow(line)
wf.close()
