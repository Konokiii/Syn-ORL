import os
import torch
import pickle
from algorithms.offline.td3_bc import preprocess_states_and_actions

for file in os.listdir('./'):
    print(file)
    try:
        if file.endswith('A.pkl'):
            with open(file, 'rb') as f:
                a = pickle.load(f)
                print(a.shape)
        if file.endswith('S.pkl'):
            with open(file, 'rb') as f:
                a, b = pickle.load(f)
                print(a.shape, b.shape)
    except Exception as e:
        print(e)