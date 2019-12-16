import os
import time
from keras import optimizers
from keras.models import load_model
from model import vggvox_model
import tools
import constants as c

def train_vggvox_model(model_load_path, model_save_path,continue_training, save_model):
    if continue_training == 1:
        print("load model from {}...".format(model_load_path))
        model = load_model(model_load_path)
    else:
        model = vggvox_model()
    audiolist, labellist = tools.get_voxceleb1_datalist(c.FA_DIR, c.TRAIN_LIST_FILE)
    train_gene = tools.DataGenerator(audiolist, labellist, c.DIM, c.MAX_SEC, c.BUCKET_STEP, c.FRAME_STEP, c.BATCH_SIZE,
                                     c.N_CLASS)

    # 编译模型
    model.compile(optimizer=optimizers.Adam(lr=c.LR),
                  loss="categorical_crossentropy",  # 使用分类交叉熵作为损失函数
                  metrics=['acc'])  # 使用精度作为指标

    # train_data["voice"] = train_data["voice"].apply(lambda x: x.reshape(1,*x.shape,1))
    # train_data["lable"] = train_data["lable"].apply(lambda x: x.reshape((1, 1, 1, 1251)))

    print("Start training...")
    history = model.fit_generator(train_gene,
                                  epochs=c.EPOCHS,
                                  steps_per_epoch=int(len(labellist) // c.BATCH_SIZE)
                                  )

    model.save_weights(filepath=c.PERSONAL_WEIGHT, overwrite=True)
    if save_model == 1:
        print("save model to {}...".format(model_save_path))
        model.save(model_save_path, overwrite=True)
    tools.draw_loss_img(history.history)
    tools.draw_acc_img(history.history)
    print("Done!")


'''
训练
'''
print("*****Check params*****\nlearn_rate:{}\nepochs:{}\nbatch_size:{}\nclass_num:{}\ncontinue_training:{}\nsave_model:{}\n*****Check params*****"
      .format(c.LR,c.EPOCHS,c.BATCH_SIZE,c.N_CLASS,c.CONTINUE_TRAINING,c.SAVE))
time.sleep(15)
train_vggvox_model(c.MODEL_LOAD_PATH, c.MODEL_SAVE_PATH, c.CONTINUE_TRAINING, c.SAVE)
