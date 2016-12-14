#!/usr/bin/env python

import os
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def cut_gaussian(x, x0, sigma=0.05, cut=8):
    prefix = 1 / sigma / np.sqrt(2*np.pi)
    gcut = prefix * np.exp(-cut**2/2)

    return prefix * np.exp(-(x-x0)**2/(2*sigma**2))

def get_bandInfo(inFile = 'OUTCAR'):
    """
    extract band energies from OUTCAR
    """

    outcar = [line for line in open(inFile) if line.strip()]

    for ii, line in enumerate(outcar):
        if 'NKPTS =' in line:
            nkpts = int(line.split()[3])
            nband = int(line.split()[-1])

        if 'ISPIN  =' in line:
            ispin = int(line.split()[2])

        if "k-points in reciprocal lattice and weights" in line:
            Lvkpts = ii + 1

        if 'reciprocal lattice vectors' in line:
            ibasis = ii + 1

        if 'E-fermi' in line:
            Efermi = float(line.split()[2])
            LineEfermi = ii + 1
            break

    # basis vector of reciprocal lattice
    B = np.array([line.split()[3:] for line in outcar[ibasis:ibasis+3]], 
                 dtype=float)
    # k-points vectors and weights
    tmp = np.array([line.split() for line in outcar[Lvkpts:Lvkpts+nkpts]],
                   dtype=float)
    vkpts = tmp[:,:3]
    wkpts = tmp[:,-1]

    # for ispin = 2, there are two extra lines "spin component..."
    N = (nband + 2) * nkpts * ispin + (ispin - 1) * 2
    bands = []
    # vkpts = []
    for line in outcar[LineEfermi:LineEfermi + N]:
        if 'spin component' in line or 'band No.' in line:
            continue
        if 'k-point' in line:
            # vkpts += [line.split()[3:]]
            continue
        bands.append(float(line.split()[1]))

    # vkpts = np.array(vkpts, dtype=float)[:nkpts]
    bands = np.array(bands, dtype=float).reshape((ispin, nkpts, nband))

    if os.path.isfile('KPOINTS'):
        kp = open('KPOINTS').readlines()

    if os.path.isfile('KPOINTS') and kp[2][0].upper() == 'L':
        Nk_in_seg = int(kp[1].strip())
        Nseg = nkpts / Nk_in_seg
        vkpt_diff = np.zeros_like(vkpts, dtype=float)
        
        for ii in range(Nseg):
            start = ii * Nk_in_seg
            end = (ii + 1) * Nk_in_seg
            vkpt_diff[start:end, :] = vkpts[start:end,:] - vkpts[start,:]

        kpt_path = np.linalg.norm(np.dot(vkpt_diff, B), axis=1)
        # kpt_path = np.sqrt(np.sum(np.dot(vkpt_diff, B)**2, axis=1))
        for ii in range(1, Nseg):
            start = ii * Nk_in_seg
            end = (ii + 1) * Nk_in_seg
            kpt_path[start:end] += kpt_path[start-1]

        kpt_path /= kpt_path[-1]
        kpt_bounds =  np.concatenate((kpt_path[0::Nk_in_seg], [1.0,]))
    else:
        # get band path
        vkpt_diff = np.diff(vkpts, axis=0)
        kpt_path = np.zeros(nkpts, dtype=float)
        kpt_path[1:] = np.cumsum(np.linalg.norm(np.dot(vkpt_diff, B), axis=1))
        kpt_path /= kpt_path[-1]

        # get boundaries of band path
        xx = np.diff(kpt_path)
        kpt_bounds = np.concatenate(([0.0,], kpt_path[np.isclose(xx, 0.0)], [1.0,]))


    return Efermi, wkpts, kpt_bounds, kpt_path, bands

################################################################################

Efermi, wkpts, kpt_bounds, kpt_path, bands = get_bandInfo()
ispin, nkpts, nband = bands.shape

# set energy zeros to Fermi energy
Ezero = Efermi

bands -= Ezero
        
################################################################################
# The Plotting part
################################################################################
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.unicode_minus'] = False

# RATIO = 0.66
#
# fig, ax = plt.subplots(nrows=1, ncols=1,
#                        sharex=False,
#                        sharey=False)
# plt.subplots_adjust(left=0.09, right=RATIO,
#                     bottom=0.10, top=0.95,
#                     wspace=0.10, hspace=0.10)
# fig.set_size_inches((6, 4))

# axDos = fig.add_axes([RATIO + 0.02, 0.10, 0.30, 0.85])
# axDos.yaxis.set_ticks_position('right')

fig = plt.figure()
fig.set_size_inches((4.0, 4.0))

ax  = plt.subplot(111)
plt.subplots_adjust(left=0.12, right=0.95,
                    bottom=0.08, top=0.95,
                    wspace=0.10, hspace=0.10)

divider = make_axes_locatable(ax)
axDos = divider.append_axes('right', size='45%', pad=0.10)

clrs = ['r', 'b']

for ii in np.arange(ispin):
    for jj in np.arange(nband):

        ax.plot(kpt_path, bands[ii,:,jj], '-o',
                ms=3, mfc=clrs[ii],mew=0.0,
                color='k', lw=1.0,
                alpha=0.6)

for bd in kpt_bounds:
    ax.axvline(x=bd, ls=':', color='k', lw=0.5, alpha=0.6)

ax.set_ylim(-5, 5)

ax.minorticks_on()
ax.tick_params(which='both', labelsize='small')

ax.set_ylabel('Energy [eV]', fontsize='small')

# pos = [0,] + list(kpt_bounds) + [1,]
# ax.set_xticks(pos)
ax.set_xticks(kpt_bounds)

kpts_name =[xx for xx in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'][:kpt_bounds.size]
# kpts_name =['M', r'$\Gamma$', 'K', 'M']
ax.set_xticklabels(kpts_name, fontsize='small')                   

################################################################################
EXTRA = 0.10
NEDOS = 1000
SIGMA = 0.05
DOS = np.zeros((NEDOS, ispin), dtype=float)

emin = bands.min()
emax = bands.max()
eran = emax - emin

emin = emin - EXTRA * eran
emax = emax + EXTRA * eran

x = np.linspace(emin, emax, NEDOS)
################################################################################
# dos generation

for sp in range(ispin):
    for kp in range(nkpts):
        for nb in range(nband):
            en = bands[sp,kp,nb]
            DOS[:,sp] += cut_gaussian(x, en, SIGMA, cut=6.0)

    axDos.plot(DOS[:,sp], x, ls='-', color=clrs[sp], lw=1.5, alpha=0.6)

axDos.set_xlabel('DOS [a.u.]', fontsize='small')

axDos.set_ylim(-5, 5)
axDos.set_xticks([])
axDos.set_yticklabels([])

axDos.minorticks_on()
axDos.tick_params(which='both', labelsize='small')

plt.savefig('kaka.png', dpi=360)
# plt.show()
