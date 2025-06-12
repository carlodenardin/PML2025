# DIRECTORY DSPRITES SETTINGS
RESULTS_DIR_DSPRITES = "results/dsprites"
CHECKPOINTS_DIR_DSPRITES = "checkpoints/dsprites"
LOGS_DIR_DSPRITES = "logs/dsprites"

# DATASET SETTINGS
URL_DSPRITES = "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"
FILENAME_DSPRITES = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
DIR_DSPRITES = "data/dsprites"

# DIRECTORY DSPRITES SETTINGS
RESULTS_DIR_MPI3D = "results/mpi3d"
CHECKPOINTS_DIR_MPI3D = "checkpoints/mpi3d"
LOGS_DIR_MPI3D = "logs/mpi3d"

# DATASET SETTINGS
URL_MPI3D = "https://huggingface.co/datasets/carlodenardin/dis/resolve/main/real3d_complicated_shapes_ordered.npz"
FILENAME_MPI3D = "real3d_complicated_shapes_ordered.npz"
DIR_MPI3D = "data/mpi3d"
SAMPLES_MPI3D = 200000

# EXPERIMENT SETTINGS
SEED = 19
BATCH_SIZE = 256
NUM_WORKERS = 12
TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]
EPOCHS = 200
PATIENCE = 10
PRECISION = "16-mixed"
MONITOR_METRIC = "val_loss"

# MODELS COMMON SETTINGS
LATENT_DIM = 10
LR_VAE = 1e-4

# BETA VAE SETTINGS
BETA = 1.0

# FACTOR VAE SETTINGS
GAMMA = 1.0
LR_DISC = 1e-4
HIDDEN_UNITS_D = 1000
NUM_LAYERS_D = 6