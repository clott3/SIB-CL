import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from glob import glob
import os

cmap = cm.get_cmap('Spectral', 6)
rdir = f"./tlogs/"

fpath = rdir + f"./identifier/"
plot_CL(fpath)
plot_FT(fpath, nsam=1000, ptep=100)
plot_testloss(fpath, nsam=1000, show_leg=False)

def plot_CL(fpath):
    ptfiles = glob(fpath+"/*.json")
    plt.figure()
    plt.subplot(121)
    plt.title(f"CL loss vs ptep")
    for testfile in ptfiles:
        # ptlr= re.findall("(?<=pt)(.*)(?=_opt)",testfile)
        epvec = []
        ervec = []
        with open(testfile) as f:
            d = json.load(f)
        for ep in d['loss']:
            epvec.append(int(ep))
            ervec.append(d['loss'][ep])
        plt.plot(epvec,ervec)
    plt.legend()
    plt.subplot(122)
    plt.title(f"TL loss vs ptep")
    for testfile in ptfiles:
        # ptlr= re.findall("(?<=pt)(.*)(?=_opt)",testfile)
        epvec = []
        ervec = []
        with open(testfile) as f:
            d = json.load(f)
        for ep in d['tl_loss']:
            epvec.append(int(ep))
            ervec.append(d['tl_loss'][ep])
        plt.plot(epvec,ervec)
    plt.legend()


def plot_FT(fpath, nsam=1000, ptep=0):
    ftfiles = glob(fpath+f"/nsam{nsam}/*.json")
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    ptep = str(ptep)
    plt.title(f"Finetuning loss at ptep {ptep} V.S. FTEP")
    epvec=[]; ervec=[];
    for testfile in ftfiles:
        # if "_"+mode+"_" not in testfile: continue
        with open(testfile) as f:
            d = json.load(f)
        try:
            for ep in d['loss'][ptep]:
                epvec.append(int(ep))
                ervec.append(d['loss'][ptep][ep])
            plt.plot(epvec,ervec)
        except KeyError as err:
            print(err, testfile)
    plt.subplot(122)
    plt.title(f"Test loss at ptep {ptep} V.S. FTEP")
    epvec=[]; ervec=[];
    pt0min, pt0fmin = 1., ""
    for testfile in ftfiles:
        # if "_"+mode+"_" not in testfile: continue
        with open(testfile) as f:
            d = json.load(f)
        try:
            for ep in d['loss'][ptep]:
                epvec.append(int(ep))
                ervec.append(d['testloss'][ptep][ep])
            minloss = np.min(np.array(ervec))
            if minloss < pt0min:
                pt0min, pt0fmin = minloss, testfile
        except KeyError as err:
            print(err, testfile)
        plt.plot(epvec,ervec)
    # print("Min from-scratch (PTEP=0) error: ", pt0min)
    # minconfig = re.findall("(?<=nsam)(.*)(?=json)",pt0fmin)[0]
    # print(minconfig)

def plot_testloss(fpath, nsam=1000,show_leg=False):
    # nsam = 1000
    ftfiles = glob(fpath+f"/nsam{nsam}/*.json")

    plt.figure()
    plt.title("Test loss V.S. PTEP")
    i=0
    gmin, ptepmin, fmin = 1., 0, ""

    for testfile in ftfiles:
        # if "_"+mode+"_" not in testfile: continue
        epvec=[]; ervec=[];
        i += 1
        if show_leg:
            ptlr= re.findall("(?<=pt)(.*)(?=_optDOS)",testfile)[0]
            ftlr= re.findall("(?<=sft)(.*)(?=_ftna)",testfile)[0]
        # if float(ftlr) != 1e-4: continue
        with open(testfile) as f:
            d = json.load(f)
        for ptep in d['testloss']:
            epvec.append(ptep)
            testvec=[];
            for ep in d['testloss'][ptep]:
                testvec.append(d['testloss'][ptep][ep])
            minloss = np.min(np.array(testvec))
            ervec.append(minloss)
            # plt.scatter(int(ptep),minloss, color=cmap(i%6))
            if minloss < gmin:
                gmin, ptepmin, fmin = minloss, int(ptep), testfile
            lab = 'pt'+ptlr+'; ft'+ftlr if show_leg else ""
        plt.plot(epvec,ervec, color=cmap(i%6), label = lab)
    if show_leg:
        plt.legend()
    print(f"Global min {gmin} @ PTEP {ptepmin}")
    print(fmin)
    # if get_minconfig:
        # minconfig = re.findall("(?<=tlogs)(.*)(?=_optDOS)",fmin)[0]
    # minconfig = re.findall("(?<=nsam)(.*)(?=json)",fmin)[0]
    # print(minconfig)
