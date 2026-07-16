"""Fig. 3 (snr_vs_ser.pdf) from results_merged.csv — same style as code/Fig/plot_snr_vs_ser.py."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
OUT_PATH = os.path.join(DATA_DIR, '../../figure/snr_vs_ser.pdf')

plt.rcParams.update({
    'font.size':       8,
    'axes.labelsize':  8,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth':  0.8,
    'grid.linewidth':  0.5,
})

color_map     = {4: '#5b9bd5', 8: '#ed7d31', 16: '#70ad47'}
linestyle_map = {'CE': '--', 'Proposed': '-'}
marker_map    = {'CE': 's',   'Proposed': 'o'}

runs = {
    'mask_nce0.01_U4':  ('Proposed', 4),
    'mask_ce_U4':       ('CE',       4),
    'mask_nce0.01_U8':  ('Proposed', 8),
    'mask_ce_U8':       ('CE',       8),
    'mask_nce0.01_U16': ('Proposed', 16),
    'mask_ce_U16':      ('CE',       16),
}

df = pd.read_csv(DATA_DIR + 'results_merged.csv')

fig, ax = plt.subplots(figsize=(3.5, 2.8))
for run, (method, U) in runs.items():
    sub = df[(df['run'] == run) & (df['SER'].astype(float) > 0)].sort_values('snr_db')
    if sub.empty:
        print(f'  (skip {run}: not in CSV yet)')
        continue
    ax.plot(sub['snr_db'], sub['SER'].astype(float),
            label=f'{method} ($U\\!=\\!{U}$)',
            color=color_map[U],
            linestyle=linestyle_map[method],
            marker=marker_map[method],
            markersize=3.5, linewidth=1.5, zorder=3)

# DE theoretical bound (scalar flat-Rayleigh QPSK reference)
snr_db  = np.arange(0, 26, 5)
snr_lin = 10 ** (snr_db / 10)
ser_de  = 1 - (0.5 * (1 + np.sqrt(snr_lin / (1 + snr_lin)))) ** 2
ax.plot(snr_db, ser_de, 'k-', linewidth=1.5, label='DE (reference)', zorder=4)

ax.set_yscale('log')
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('SER')
ax.set_xlim([0, 25])
ax.set_ylim([3e-8, 5.0])   # headroom above SER=1 so the legend clears the curves
ax.grid(True, which='both', alpha=0.25, linestyle='--')
ax.tick_params(which='both', direction='in')
ax.legend(loc='upper right', ncol=4, fontsize=5.0, handlelength=0.7,
          handletextpad=0.2, columnspacing=0.25, borderpad=0.25,
          labelspacing=0.25, framealpha=0.9, edgecolor='lightgray')
fig.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.97)
plt.savefig(OUT_PATH, dpi=300)
print(f'Saved: {OUT_PATH}')
