from eta00.threshold00history_all import data as data0000
from eta00.threshold03history_all import data as data0300
from eta00.threshold04history_all import data as data0400
from eta00.threshold05history_all import data as data0500

from eta05.threshold00history_all import data as data0005
from eta05.threshold03history_all import data as data0305
from eta05.threshold04history_all import data as data0405
from eta05.threshold05history_all import data as data0505

from eta10.threshold00history_all import data as data0010
from eta10.threshold03history_all import data as data0310
from eta10.threshold04history_all import data as data0410
from eta10.threshold05history_all import data as data0510

# from eta00opt.threshold00history_all import data as data0000
# from eta00opt.threshold03history_all import data as data0300
# from eta00opt.threshold04history_all import data as data0400
# from eta00opt.threshold05history_all import data as data0500

# from eta05opt.threshold00history_all import data as data0005
# from eta05.threshold03history_all import data as data0305
# from eta05.threshold04history_all import data as data0405
# from eta05.threshold05history_all import data as data0505

# from eta10opt.threshold00history_all import data as data0010
# from eta10opt.threshold03history_all import data as data0310
# from eta10opt.threshold04history_all import data as data0410
# from eta10opt.threshold05history_all import data as data0510

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

data00 = [data0000, data0300, data0400, data0500]
data05 = [data0005, data0305, data0405, data0505]
data10 = [data0010, data0310, data0410, data0510]

datas = [data00, data05, data10]
figNames = ["eta = 0.0", "eta = 0.5", "eta = 1.0"]
names = ["Threshold 0.0", "Threshold 0.3", "Threshold 0.4", "Threshold 0.5"]
plt.rcParams.update({'font.size': 10})

fig, axs = plt.subplots(3, 4, figsize=(8,11))
for ind, data in enumerate(datas):
    dfs = [pd.DataFrame(d) for d in data]
    bss = list(set(dfs[0].bs))
    lrs = list(set(dfs[0].lr))
    drs = list(set(dfs[0].dr))
    markers = ['o', '^', 'P']
    colors = ["red", "orange", "yellow", "green"]
    edgecolors = ["blue", "pink", "black", "grey"]
    for i in range(len(axs[ind])):
        for _, row in dfs[i].iterrows():
            axs[ind][i].scatter(row.pl//1000, math.log(row.total_cost), c=colors[bss.index(row.bs)], marker=markers[lrs.index(row.lr)], edgecolors=edgecolors[drs.index(row.dr)], s=80)

        f = lambda m,c,e: plt.scatter([],[],marker=m, color=c, edgecolors=e)

        # plt.ylim(13.5, 13.7)
        handles = [f("o", colors[i], "white") for i in range(len(colors))]
        handles += [f(markers[i], "black", "black") for i in range(len(markers))]
        handles += [f("o", "white", edgecolors[i]) for i in range(len(edgecolors))]
        
        axs[ind][i].set_xticks(range(100, 176, 25), ["100", "125", "150", "175"])
        if i == 0:
            axs[ind][i].set_ylabel(figNames[ind])

        if ind == 2:
            axs[ind][i].set_xlabel(names[i])
        # plt.title("Log of profiling costs [eta = 1, threshold = 0.4]")

fig.text(0.5, 0.02, 'Power Limit', ha='center', fontsize=12)
fig.supylabel('Log Cost')
fig.legend(handles, [f"BS {bs}" for bs in bss] + [f"LR {lr}" for lr in lrs] + [f"DR {dr}" for dr in drs], ncol=3, loc='upper left', framealpha=1, prop={'size': 9})
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle(f"Profiling results")
plt.show()