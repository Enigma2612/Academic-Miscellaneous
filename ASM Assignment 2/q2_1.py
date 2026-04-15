import math
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))

data = {
    'E1': 140e9,
    'E2': 10e9,
    'G12': 7e9,
    'v12': 0.3,
    'v23': 0.2
}

data['v21'] = data['v12'] * data['E2'] / data['E1']


c = lambda x,n : (math.cos(math.radians(x)))**n
s = lambda x,n : (math.sin(math.radians(x)))**n

def get_new_S(S, x):   #x -> angle in degrees
    S11 = S[0][0] * c(x, 4) + S[1][1] * s(x,4) + (S[2][2] + 2 * S[0][1]) * c(x,2) * s(x,2)
    S22 = S[0][0] * s(x, 4) + S[1][1] * c(x,4) + (S[2][2] + 2 * S[0][1]) * c(x,2) * s(x,2)
    S33 = (2 * S[0][0] + 2 * S[1][1] - S[2][2] - 4 * S[0][1]) * 2 * c(x,2) * s(x,2) + S[2][2] * (c(x,4) + s(x,4))
    S12 = (S[0][0] + S[1][1] - S[2][2]) * c(x,2) * s(x,2) + S[0][1] * (c(x,4) + s(x,4))
    S13 = (2 * S[0][0] - S[2][2] - 2 * S[0][1]) * c(x,3) * s(x,1) - (2 * S[1][1] - S[2][2] - 2 * S[0][1]) * c(x,1) * s(x,3)
    S23 = (2 * S[0][0] - S[2][2] - 2 * S[0][1]) * c(x,1) * s(x,3) - (2 * S[1][1] - S[2][2] - 2 * S[0][1]) * c(x,3) * s(x,1)
    S21 = S12
    S31 = S13
    S32 = S23

    return [[S11, S12, S13], [S21, S22, S23], [S31, S32, S33]]

def get_vals(x):
    return [1/x[0][0] * 1e-9, 1/x[1][1] * 1e-9, 1/x[2][2] * 1e-9, -x[1][0] / x[0][0], x[0][2]/x[2][2], x[1][2] / x[2][2]]

def save_data(S_vals):
    data = []
    for x in S_vals:
        col = [x.pop(0)]
        col += [1/x[0][0] * 1e-9, 1/x[1][1] * 1e-9, 1/x[2][2] * 1e-9, -x[1][0] / x[0][0], x[0][2]/x[2][2], x[1][2] / x[2][2]]

        col = [round(i, 3) for i in col]
        data.append(col)

    

    with open(path + '/q2_1_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Ø", "Ex (GPa)", "Ey (GPa)", "Gxy (GPa)", "vxy", "nxy,x", "nxy,y"])
        writer.writerows(data)
        print("saved")
    return data

S = [[1/data['E1'], -data['v21']/data['E2'], 0],
     [-data['v12']/data['E1'], 1/data['E2'], 0],
     [0, 0, 1/data['G12']]]

S_vals = []
angles = [0, 20, 30, 45, 60, 70, 90]

for angle in angles:
    S_vals.append([angle] + get_new_S(S, angle))


full_data = []
ref_lis = np.linspace(0,180,1000)

for i in ref_lis:
    full_data.append(get_vals(get_new_S(S, i)))

full_data = np.linalg.matrix_transpose(np.matrix(full_data)).tolist()


import matplotlib.pyplot as plt

def setup_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid')  # clean base
    
    plt.rcParams.update({
        # ===== Figure =====
        "figure.figsize": (10, 6),
        "figure.dpi": 120,
        
        # ===== Fonts =====
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        
        # ===== Lines =====
        "lines.linewidth": 2.5,
        
        # ===== Grid =====
        "grid.alpha": 0.3,
        
        # ===== Legend =====
        "legend.fontsize": 12,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": True,
        
        # ===== Axes =====
        # "axes.spines.top": False,
        # "axes.spines.right": False,
    })

setup_plot_style()

colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
]

ylabels = ['Eₓ (GPa)', 'Eᵧ (GPa)', 'Gₓᵧ (GPa)', 'νₓᵧ', 'ηₓᵧ,ₓ', 'ηₓᵧ,ᵧ']
titles = [  "Young's Modulus Eₓ vs Fibre Orientation",
            "Young's Modulus Eᵧ vs Fibre Orientation",
            "Shear Modulus Gₓᵧ vs Fibre Orientation",
            "Poisson's Ratio νₓᵧ vs Fibre Orientation",
            "Mutual Influence ηₓᵧ,ₓ vs Fibre Orientation",
            "Mutual Influence ηₓᵧ,ᵧ vs Fibre Orientation",
          ]
xlabel = 'Fibre Orientation θ (in degrees)'


fig, axes = plt.subplots(nrows = len(full_data), ncols=1, figsize=(10, 18))

if not isinstance(axes, (list, np.ndarray)):
    axes = [axes]

for i in range(0, len(full_data)):
    ax = axes[i]

    ax.plot(ref_lis, full_data[i], color=colors[i])

    ax.set_ylabel(ylabels[i])
    ax.set_title(titles[i])

    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_xlabel(xlabel)

plt.tight_layout()
plt.subplots_adjust(hspace=1)

plt.savefig(path + '/graph_2_1', dpi=600, bbox_inches='tight')
plt.show()