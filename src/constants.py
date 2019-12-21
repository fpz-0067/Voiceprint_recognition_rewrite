
# Signal processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512 # FFT（快速傅立叶变换）是指通过在计算项中使用对称性可以有效地计算离散傅立叶变换（DFT）的方法。当n为2的幂时，对称性最高，因此，对于这些大小，变换效率最高。
BUCKET_STEP = 1
MAX_SEC = 10
DIM = (512, 300, 1)

# Model
WEIGHTS_FILE = "weights/weights.h5"
COST_METRIC = "cosine"  # euclidean or cosine
INPUT_SHAPE=(NUM_FFT,None,1)

# IO on windows
# FA_DIR = "F:/Vox_data/vox1_dev_wav/wav/"
# TRAIN_LIST_FILE = "cfg/trainlist.txt"
# PERSONAL_WEIGHT = "D:\Python_projects/vggvox_rewrite\weights/weight_128.h5"

# IO on linux
FA_DIR = "/home/longfuhui/all_data/vox1-dev-wav/wav/"
TRAIN_LIST_FILE = "cfg/trainlist.txt"
PERSONAL_WEIGHT = "weights/weight_128.h5"

ENROLL_LIST_FILE = "cfg/new_enroll_list.csv"
TEST_LIST_FILE = "cfg/new_test_list.csv"
RESULT_FILE = "results/results.csv"

# train
MODEL_LOAD_PATH = "models/model_128.h5"
MODEL_SAVE_PATH = "models/model_128.h5"

MODEL_FA_PATH = "models/m_128/"
TENSORBOARD_LOG_PATH = "tensorboard/log"
LOSS_PNG = "img/loss.png"
ACC_PNG = "img/acc.png"
CONTINUE_TRAINING = 0
SAVE = 1
LR = 0.001
EPOCHS = 55
BATCH_SIZE = 8
N_CLASS = 128