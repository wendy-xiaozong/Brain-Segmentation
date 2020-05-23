import tensorflow as tf
import os
from random import randint
import numpy as np
from time import time

from utils.data import get_dataset
from utils.model_3D_Patchwise_Unet import build_model
from utils.losses import get_loss, dice_coefficient

FLAGS = tf.compat.v1.flags.FLAGS  # to get argument from cmd


def train(patch_size):
    dataset = get_dataset()

    pz = patch_size[0]  # "8,24,24"
    py = patch_size[1]
    px = patch_size[2]

    net = build_model()
    net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
                loss=get_loss, metrics=[dice_coefficient])
    net.summary()

    # new_rate = FLAGS.learning_rate
    # saver = tf.train.Saver(keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint)
    #
    # try:
    #     checkpoints = FLAGS.test_checkpoints.split(",")
    #     saver.restore(sess, FLAGS.checkpoint_path + '-' + checkpoints[0])
    #     start_step = int(checkpoints[0])
    #     rate_updates = int(start_step / FLAGS.steps_to_learning_rate_update)
    #     for n in range(rate_updates):
    #         new_rate = new_rate * (1 - FLAGS.learning_rate_decrease)
    #     print("\nCheckpoint {} loaded\n".format(checkpoints[0]))
    # except:
    #     start_step = 0
    #     print("\nNew Initialization\n")
    #     pass
    #
    # print()
    # for n in range(start_step, FLAGS.max_steps):
    #     # t1 = time()
    #     data = get_next_train_batch(dataset, patch_size)
    #     # t2 = time()
    #     _, s_loss, s_dsc = sess.run((train,
    #                                  net["loss"],
    #                                  net["dsc"]),
    #                                 feed_dict={x_flair: data["flair"],
    #                                            x_t1: data["t1"],
    #                                            x_ir: data["ir"],
    #                                            y_gt: data["label"],
    #                                            rate: new_rate})
    #     # t3 = time()
    #     train_status = 'loss: {:0.4f} - bgr:{:0.3f} gm:{:0.3f} bg:{:0.3f} ' + \
    #                    'wm:{:0.3f} ' + \
    #                    'wmh:{:0.3f} cf:{:0.3f} ce:{:0.3f} ve:{:0.3f} bs:{:0.3f} if:{:0.3f} ' + \
    #                    'ot:{:0.3f} - step: {}/{}'
    #     print(train_status.format(s_loss, s_dsc[0], s_dsc[1], s_dsc[2], s_dsc[3],
    #                               s_dsc[4], s_dsc[5], s_dsc[6], s_dsc[7], s_dsc[8], s_dsc[9], s_dsc[10],
    #                               n + 1, FLAGS.max_steps))
    #     # print("data {:0.4f}, model {:0.4f}".format(t2-t1,t3-t2))
    #
    #     if (n + 1) % FLAGS.steps_to_val == 0:
    #         print()
    #         val_size = len(dataset["val"])
    #         for m in range(val_size):
    #             subject = dataset["val"][m]
    #             subject.new_prediction()
    #             shape = subject.shape
    #             for z in range(0, shape[1] - pz, int(pz / 2)):
    #                 for y in range(0, int(py / 2) + 1, int(py / 2)):
    #                     y2 = y + int((shape[2] - y) / py) * py
    #                     for x in range(0, int(px / 2) + 1, int(px / 2)):
    #                         x2 = x + int((shape[3] - x) / px) * px
    #                         tmp_flair = subject.flair_array[:, z:z + pz, y:y2, x:x2]
    #                         tmp_shape = list(tmp_flair.shape)
    #                         tmp_flair = re_arrange_array(tmp_flair, tmp_shape, "input")
    #                         tmp_t1 = subject.t1_array[:, z:z + pz, y:y2, x:x2]
    #                         tmp_t1 = re_arrange_array(tmp_t1, tmp_shape, "input")
    #                         tmp_ir = subject.ir_array[:, z:z + pz, y:y2, x:x2]
    #                         tmp_ir = re_arrange_array(tmp_ir, tmp_shape, "input")
    #                         tmp_label = sess.run(net["output"], feed_dict={ \
    #                             x_flair: tmp_flair,
    #                             x_t1: tmp_t1,
    #                             x_ir: tmp_ir})
    #                         tmp_shape[-1] = subject.pred_array.shape[-1]
    #                         tmp_label = re_arrange_array(tmp_label, tmp_shape, "output")
    #                         subject.pred_array[:, z:z + pz, y:y2, x:x2] += tmp_label
    #             subject.pred_array = subject.pred_array / 8
    #             subject.get_dsc()
    #             test_status = "{}\nBackground:\t\t{:0.4f}\n" + \
    #                           "Cortical gray matter:\t{:0.4f}\n" + \
    #                           "Basal ganglia:\t\t{:0.4f}\n" + \
    #                           "White matter:\t\t{:0.4f}\n" + \
    #                           "White matter lesions:\t{:0.4f}\n" + \
    #                           "Cerebrospinal fluid:\t{:0.4f}\n" + \
    #                           "Ventricles:\t\t{:0.4f}\n" + \
    #                           "Cerebellum:\t\t{:0.4f}\n" + \
    #                           "Brain stem:\t\t{:0.4f}\n" + \
    #                           "Infarction:\t\t{:0.4f}\n" + \
    #                           "Other:\t\t\t{:0.4f}\n"
    #             print(test_status.format(subject.name, subject.dsc[0],
    #                                      subject.dsc[1],
    #                                      subject.dsc[2],
    #                                      subject.dsc[3],
    #                                      subject.dsc[4],
    #                                      subject.dsc[5],
    #                                      subject.dsc[6],
    #                                      subject.dsc[7],
    #                                      subject.dsc[8],
    #                                      subject.dsc[9],
    #                                      subject.dsc[10]))
    #             del subject.pred_array
    #         print()
    #
    #     if (n + 1) % FLAGS.steps_to_learning_rate_update == 0:
    #         new_rate = new_rate * (1 - FLAGS.learning_rate_decrease)
    #         print('New learning rate = {}'.format(new_rate))
    #         print()
    #
    #     if (n + 1) % FLAGS.steps_to_save_checkpoint == 0:
    #         saver.save(sess,
    #                    FLAGS.checkpoint_path,
    #                    global_step=n + 1,
    #                    write_meta_graph=False)
    #         print('Check point saved')
    #         print()


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device
    patch_size = list(map(int, FLAGS.patch_size.split(",")))  # patch_size = "8,24,24"
    train(patch_size=patch_size)
    print("done!")


if __name__ == '__main__':
    main()
