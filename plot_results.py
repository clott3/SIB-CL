import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from glob import glob
import os

def get_best_ptep(fpath, ignorensam = [], fixep='', addfpath=""):
    gmin = {}
    err = {}
    flist = glob(fpath) if addfpath == "" else glob(fpath)+glob(addfpath)
    for file1 in flist:
        with open(file1) as f:
            d = json.load(f)
        for nsam,v in d.items():
            if int(nsam) in ignorensam: continue
            if int(nsam) not in err:
                err[int(nsam)] = []
                gmin[int(nsam)] = 1.
            minmean = 1.
            for ep, v2 in v.items():
                if fixep != '' and str(ep) == fixep: continue
                errs = []
                for seed in v2:
                    errs.append(v2[seed][0])
                mean1 = np.mean(np.array(errs))
                std1 = np.std(np.array(errs))
                if mean1 < minmean:
                    minmean = mean1
                    minstd = std1
                    minep = ep
            if minmean < gmin[int(nsam)]:
                err[int(nsam)] = [minmean, minstd, minep]
                gmin[int(nsam)] = minmean
    lists = sorted(err.items()) # sorted by key, return a list of tuples
    if len(lists) != 0:
        x, y = zip(*lists) # unpack a list of pairs into two tuples
    return x,y

fs_fpath = "./dicts/*FS*/*.json"
tl_fpath = "./dicts/*tl*/*.json"
sibcl_fpath = "./dicts/*sibcl*/*.json"

pltdicts = [fs_fpath, tl_fpath, sibcl_fpath]
plt.figure()
ax = plt.axes()
cmap = cm.get_cmap('inferno',5)
ic = 1
for file in pltdicts:
    x,y = get_best_ptep(file, ignorensam = [])
    y=np.array(y, dtype=float)
    plt.errorbar(x,y[:,0],yerr=y[:,1],color=cmap(ic), marker='o', ms=3,capsize=2, elinewidth=2)
    print(y[:,2])
    ic+=1
    print(y[:,0])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([50,100,200,400,800,1600,3200])
ax.set_xticklabels(['50','100','200','400','800','1600','3200'],minor=False)
ax.set_yticks([2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2])
ax.set_yticklabels(['2','3','4','5','6','7','8'],minor=False)
# plt.savefig("phc.svg")
