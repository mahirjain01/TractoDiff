import os
import pickle 

with open('/tracto/TractoDiff/output/AF_L/sub-1030.pkl', 'rb') as file:
    data = pickle.load(file)

print((data[2]['observations'].shape))