import math
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

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

def save_data(S_vals):
    data = []
    for x in S_vals:
        col = [x.pop(0)]
        col += [1/x[0][0] * 1e-9, 1/x[1][1] * 1e-9, 1/x[2][2] * 1e-9, -x[1][0] / x[0][0], x[0][2]/x[2][2], x[1][2] / x[2][2]]

        col = [round(i, 3) for i in col]
        data.append(col)

    path = os.path.dirname(os.path.abspath(__file__))

    with open(path + '/q2_1_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Ø", "Ex (GPa)", "Ey (GPa)", "Gxy (GPa)", "vxy", "nxy,x", "nxy,y"])
        writer.writerows(data)
        print("saved")

S = [[1/data['E1'], -data['v21']/data['E2'], 0],
     [-data['v12']/data['E1'], 1/data['E2'], 0],
     [0, 0, 1/data['G12']]]

C = np.linalg.inv(np.matrix(S))
print(C)
