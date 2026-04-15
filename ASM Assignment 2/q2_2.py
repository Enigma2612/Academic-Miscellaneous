import math
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))


#Given Data

data = {
    'E1': 140e9,
    'E2': 10e9,
    'G12': 7e9,
    'v12': 0.3,
    'v23': 0.2
}

data['v21'] = data['v12'] * data['E2'] / data['E1']

t = 1e-3 #1mm


#Functions

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

def get_new_C(S, x):
    return np.linalg.inv(np.matrix(get_new_S(S,x)))


def A(ply):
    z = len(ply)
    A = np.zeros((3,3))
    for i in range(0, z):
        A += t * C_matrices[ply[i]]
    
    return np.round(A/1e6,3)

def B(ply):
    B = np.zeros((3,3))
    mid = len(ply)/2 * t
    z = [(i*t) - mid for i in range(len(ply)+1)]

    for i in range(len(ply)):
        B += (z[i+1]**2 - z[i]**2)/2 * C_matrices[ply[i]]
    
    return np.round(-B,3)
    

def D(ply):
    D = np.zeros((3,3))
    mid = len(ply)/2 * t
    z = [(i*t) - mid for i in range(len(ply)+1)]

    for i in range(len(ply)):
        D += (z[i+1]**3 - z[i]**3)/3 * C_matrices[ply[i]]
    
    return np.round(D,3)
    

#Data 

S = [[1/data['E1'], -data['v21']/data['E2'], 0],
     [-data['v12']/data['E1'], 1/data['E2'], 0],
     [0, 0, 1/data['G12']]]

C = np.linalg.inv(np.matrix(S))

plies = [
    [90, 60, 30, 30, 60, 90],
    [90, 60, -60, 30, -30],
    [-45, 0, 45, 90],
    [-108, -72, -36, 0, 36]
]

all_angles = []
for i in plies:
    all_angles.extend(i)
all_angles = list(set(all_angles))

C_matrices = {}

for angle in all_angles:
    C_matrices[angle] = get_new_C(S, angle)

from pprint import pprint

for ply in plies:
    print(ply)
    print("\nA\n")
    pprint(A(ply))
    print("\nB\n")
    pprint(B(ply))
    print("\nD\n")
    pprint(D(ply))
    print("\n" + "-"*40)