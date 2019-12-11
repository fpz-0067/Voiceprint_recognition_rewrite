import os
import time
from keras import optimizers
from model import vggvox_model
import tools
import constants as c

# 方法：训练
def train_vggvox_model(train_list_file):
    model = vggvox_model()
    audiolist, labellist = tools.get_voxceleb1_datalist(c.FA_DIR,c.TRAIN_LIST_FILE)

    train_gene = tools.DataGenerator(audiolist, labellist,c.DIM,c.MAX_SEC,c.BUCKET_STEP,c.FRAME_STEP,c.BATCH_SIZE,c.N_CLASS)

    # 编译模型
    model.compile(optimizer=optimizers.Adam(lr=0.04),
                  loss="categorical_crossentropy",# 使用分类交叉熵作为损失函数
                  metrics=['acc'])  # 使用精度作为指标

    # train_data["voice"] = train_data["voice"].apply(lambda x: x.reshape(1,*x.shape,1))
    # train_data["lable"] = train_data["lable"].apply(lambda x: x.reshape((1, 1, 1, 1251)))

    print("Start training...")
    history = model.fit_generator(train_gene,
                                  epochs=c.EPOCHS,
                                  steps_per_epoch=int(len(labellist)//c.BATCH_SIZE)
                                  )

    model.save_weights(filepath=c.PERSONAL_WEIGHT)
    tools.draw_loss_img(history.history)
    tools.draw_acc_img(history.history)
    print("Done!")

# 测试方法：训练
train_vggvox_model(c.TRAIN_LIST_FILE)