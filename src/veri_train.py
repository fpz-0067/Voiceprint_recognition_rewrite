import os
import time
import keras
from keras import optimizers
from keras.models import load_model
import keras.backend as K
from model import vggvox_model
import tools
import constants as c


def train_vggvox_model(model_load_path, model_save_path,continue_training, save_model):
    audiolist, labellist = tools.get_voxceleb1_datalist(c.FA_DIR, c.VERI_TRAIN_LIST_FILE)
    train_gene = tools.DataGenerator(audiolist, labellist, c.DIM, c.MAX_SEC, c.BUCKET_STEP, c.FRAME_STEP, c.BATCH_SIZE,
                                     c.N_CLASS)
    if continue_training == 1:
        print("load model from {}...".format(model_load_path))
        model = load_model(model_load_path)
    else:
        model = vggvox_model()
        # 编译模型
        model.compile(optimizer=optimizers.Adam(lr=c.LR,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                      loss="categorical_crossentropy",  # 使用分类交叉熵作为损失函数
                      metrics=['acc'])  # 使用精度作为指标

    # train_data["voice"] = train_data["voice"].apply(lambda x: x.reshape(1,*x.shape,1))
    # train_data["lable"] = train_data["lable"].apply(lambda x: x.reshape((1, 1, 1, 1251)))

    tbcallbacks = keras.callbacks.TensorBoard(log_dir=c.TENSORBOARD_LOG_PATH, histogram_freq=0, write_graph=True, write_images=False,
                                              update_freq=c.BATCH_SIZE * 10000)
    callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(c.VERI_MODEL_FA_PATH,'veri_model_128_{epoch:02d}_{loss:.3f}_{acc:.3f}.h5'),
                                                 monitor='loss',
                                                 mode='min',
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 period=5),
                 tbcallbacks]

    print("Start training...")
    history = model.fit_generator(train_gene,
                                  epochs=c.EPOCHS,
                                  steps_per_epoch=int(len(labellist) // c.BATCH_SIZE),
                                  callbacks=callbacks
                                  )

    print("save weights to {}...".format(c.PERSONAL_WEIGHT))
    model.save_weights(filepath=c.PERSONAL_WEIGHT, overwrite=True)
    if save_model == 1:
        print("save model to {}...".format(model_save_path))
        model.save(model_save_path, overwrite=True)
    tools.draw_loss_img(history.history,c.LOSS_PNG)
    tools.draw_acc_img(history.history,c.ACC_PNG)
    print("Done!")

'''
训练
'''
print("*****Check params*****\nlearn_rate:{}\nepochs:{}\nbatch_size:{}\nclass_num:{}\ncontinue_training:{}\nsave_model:{}\n*****Check params*****"
      .format(c.LR,c.EPOCHS,c.BATCH_SIZE,c.N_CLASS,c.CONTINUE_TRAINING,c.SAVE))
time.sleep(15)
# set_learning_phase(0)
train_vggvox_model(c.VERI_MODEL_PATH, c.VERI_MODEL_PATH, c.CONTINUE_TRAINING, c.SAVE)
