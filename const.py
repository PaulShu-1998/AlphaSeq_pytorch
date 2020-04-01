import torch

# ####CONFIG
# CUDA varibale from Torch
CUDA = torch.cuda.is_available()
# Dtype of the tensors depending on CUDA
DEVICE = torch.device("cuda") if CUDA else torch.device("cpu")
# Number of self-play parallel games
PARALLEL_SELF_PLAY = 2
# Number of evaluation parallel games
PARALLEL_EVAL = 3
# MCTS parallel
MCTS_PARALLEL = 4

# ####GLOBAL
# Length of the Squence
N = 5
# Number of the Squence
M = 12
# number of root
q = 4
# Learning rate
LR = 0.01
# Maximum ratio that can be replaced in the rotation buffer
MAX_REPLACEMENT = 0.4
# Number of MCTS simulation
MCTS_SIM = 64
# Exploration constant
C_PUCT = 0.2
# L2 Regularization
L2_REG = 0.0001
# Momentum
MOMENTUM = 0.9
# Activate MCTS
MCTS_FLAG = True
# Epsilon for Dirichlet noise
EPS = 0.25
# Alpha for Dirichlet noise
ALPHA = 0.03
# Batch size for evaluation during MCTS
BATCH_SIZE_EVAL = 2
# Number of self-play before training
SELF_PLAY_MATCH = PARALLEL_SELF_PLAY
# Number of moves before changing temperature to stop exploration
TEMPERATURE_MOVE = 5

# ####TRAINING
# #Number of moves to consider when creating the batch
MOVES = 2000
# #Number of mini-batch before evaluation during training
BATCH_SIZE = 64
# #Number of channels of the output feature maps
OUTPLANES_MAP = 10
# #Shape of the input state
INPLANES = 2
# #Probabilities for all moves + pass
OUTPLANES = q ** N
# #Number of residual blocks
BLOCKS = 10
# #Number of hidden layer in policy net
A2C = 3
# #Number of training step before evaluating
TRAIN_STEPS = 6 * BATCH_SIZE
# #Optimizer
ADAM = False
# #Learning rate annealing factor
LR_DECAY = 0.1
# #Learning rate annealing interval
LR_DECAY_ITE = 100 * TRAIN_STEPS
# #Print the loss
LOSS_TICK = BATCH_SIZE // 4
# #Refresh the dataset
REFRESH_TICK = BATCH_SIZE

# ####EVALUATION
# #Number of matches against its old version to evaluate the newly trained network
EVAL_MATCHES = 20
# #Threshold to keep the new neural net
EVAL_THRESHOLD = 1.05

