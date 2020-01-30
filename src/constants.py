
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
INPUT_SHAPE=(NUM_FFT,300,1)

# IO on windows
# FA_DIR = "F:/Vox_data/vox1_dev_wav/wav/"
# TRAIN_LIST_FILE = "a_veri/trainlist.txt"
# PERSONAL_WEIGHT = "D:\Python_projects/vggvox_rewrite\weights/weight_128.h5"

# IO on linux
FA_DIR = "/home/longfuhui/all_data/vox1-dev-wav/wav/"
PERSONAL_WEIGHT = "weights/weight_128.h5"
RESULT_FILE = "results/results.csv"

# train
TENSORBOARD_LOG_PATH = "tensorboard/log"
LOSS_PNG = "img/loss.png"
ACC_PNG = "img/acc.png"
CONTINUE_TRAINING = 1
SAVE = 1
LR = 0.001
EPOCHS = 20
BATCH_SIZE = 64
N_CLASS = 128

'''
Identification
'''
# train
IDEN_TRAIN_LIST_FILE = "a_iden/train_list_iden.txt"
IDEN_MODEL_FA_PATH = "models/iden/m_128/"
IDEN_MODEL_PATH = "models/iden/m_128/iden_model_128_20_0.017_1.000_add75.h5"

# test
IDEN_TEST_FILE = "a_iden/test_for_iden.txt"
IDEN_MODEL_LOAD_PATH = "models/iden/iden_model_test.h5"

'''
verification
'''
# train
VERI_TRAIN_LIST_FILE = "a_veri/train_list_veri.txt"
VERI_MODEL_FA_PATH = "models/veri/m_128/"
VERI_MODEL_PATH = "models/veri/veri_model_test.h5"

# test
VERI_TEST_FILE = "a_veri/voxceleb1_veri_test.txt"
VERI_MODEL_LOAD_PATH = "models/veri/veri_model_test.h5"