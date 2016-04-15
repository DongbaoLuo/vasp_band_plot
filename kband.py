#!/usr/bin/env python

import os
import numpy as np

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

        if 'E-fermi' in line:
            Efermi = float(line.split()[2])
            LineEfermi = ii + 1

        if 'reciprocal lattice vectors' in line:
            ibasis = ii + 1

    # basis vector of reciprocal lattice
    B = np.array([line.split()[3:] for line in outcar[ibasis:ibasis+3]], 
                 dtype=float)

    # print nkpts, nband, ispin, Efermi
    # print B

    # for ispin = 2, there are two extra lines "spin component..."
    N = (nband + 2) * nkpts * ispin + (ispin - 1) * 2
    bands = []
    vkpts = []
    for line in outcar[LineEfermi:LineEfermi + N]:
        if 'spin component' in line or 'band No.' in line:
            continue
        if 'k-point' in line:
            vkpts += [line.split()[3:]]
        else:
            bands.append(float(line.split()[1]))

    vkpts = np.array(vkpts, dtype=float)[:nkpts]
    bands = np.array(bands, dtype=float).reshape((ispin, nkpts, nband))

    # get band path
    vkpt_diff = np.diff(vkpts, axis=0)
    kpt_path = np.zeros(nkpts, dtype=float)
    kpt_path[1:] = np.cumsum(np.linalg.norm(np.dot(vkpt_diff, B), axis=1))
    kpt_path /= kpt_path[-1]

    # get boundaries of band path
    xx = np.diff(kpt_path)
    kpt_bounds = kpt_path[np.isclose(xx, 0.0)]

    # print kpt_bounds
    # print kpt_path.size, bands.shape

    return Efermi, kpt_bounds, kpt_path, bands

################################################################################

Efermi, kpt_bounds, kpt_path, bands = get_bandInfo()
ispin, nkpts, nbands = bands.shape

# set energy zeros to Fermi energy
bands -= Efermi
        
################################################################################
# The Plotting part
################################################################################
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.unicode_minus'] = False

RATIO = 0.95

fig, ax = plt.subplots(nrows=1, ncols=1,
                       sharex=False,
                       sharey=False)
plt.subplots_adjust(left=0.12, right=RATIO,
                    bottom=0.08, top=0.95,
                    wspace=0.10, hspace=0.10)
fig.set_size_inches((4, 4))

# axDos = fig.add_axes([RATIO + 0.03, 0.10, 0.3, 0.85])
# axDos.yaxis.set_ticks_position('right')

clrs = ['r', 'b']

for ii in np.arange(ispin):
    for jj in np.arange(nbands):

        ax.plot(kpt_path, bands[ii,:,jj], '-o',
                ms=3, mfc=clrs[ii],mew=0.5,
                color='k', lw=1.0,
                alpha=0.6)

for bd in kpt_bounds:
    ax.axvline(x=bd, ls='--', color='k', lw=0.5, alpha=0.6)

ax.set_ylim(-2.5, 2.5)

ax.minorticks_on()
ax.tick_params(which='both', labelsize='small')

ax.set_ylabel('Energy [eV]', fontsize='medium')

pos = [0,] + list(kpt_bounds) + [1,]
ax.set_xticks(pos)

kpts_name =[xx for xx in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'][:len(pos)]
ax.set_xticklabels(kpts_name, pos, fontsize='small')                   

################################################################################
# axDos.set_ylim(-2, 2)
# axDos.set_yticklabels([])
#
# axDos.minorticks_on()
# axDos.tick_params(which='both', labelsize='small')

plt.savefig('kaka.png', dpi=360)
# plt.show()
