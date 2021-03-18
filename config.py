import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED=10
FC1_UNITS=400
FC2_UNITS=300
GAMMA = 0.995
TAU = 1e-3              
LR_ACTOR = 1e-4         
LR_CRITIC = 1e-3        
WEIGHT_DECAY = 0.       
ADD_OU_NOISE=True
MU = 0.0
THETA = 0.15
SIGMA = 0.2
BUFFER_SIZE=int(1e5)
BATCH_SIZE=200
UPDATE_EVERY=4
MULTIPLE_LEARN_PER_UPDATE=3
NOISE=1.0