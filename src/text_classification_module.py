# -*- coding: UTF-8 -*- 
'''
Created on Wed May 6 15:23:56 2020

@author: Damon Li
'''

import tensorflow as tf
from text_cnn_module import TextCNN
from utils import text_cls_data_loader
import jieba
import os

class text_classifier(object):

    def __init__(self, tf_session, setting):

        with tf_session.as_default():
            with tf_session.graph.as_default(): # 定义属于计算图graph的张量和操作
                latest_save_time = max([int(dir) for dir in os.listdir(setting.text_cnn_model_save_dir)])

                latest_checkpoint_dir = os.path.join(setting.text_cnn_model_save_dir, str(latest_save_time))

                self.text_cls_dropout_prob = setting.text_cls_dropout_prob #1.0

                self.text_cls_data_loader = text_cls_data_loader(setting)

                self.text_cls_data_loader.load_embedding()

                self.text_cnn = TextCNN(embedding_dir=setting.text_cls_embedding_dir,
                                        sequence_length=setting.text_cls_sentence_len, #20
                                        num_classes=setting.num_classes,#9
                                        filter_sizes=list(map(int, setting.filter_sizes.split(" "))),#[2,3,4]
                                        num_filters=setting.num_filters,#128
                                        vocab_processor_dir=os.path.join(latest_checkpoint_dir, 'vocab'),
                                        l2_reg_lambda=setting.l2_lambda,#0
                                        is_inference=True,
                                        device_id=setting.cpu_id)

                self.text_cnn.build_model()

                self.saver = tf.train.Saver(max_to_keep=setting.text_cls_num_checkpoints)
                ckpt       = tf.train.get_checkpoint_state(os.path.join(latest_checkpoint_dir, 'checkpoints'))

                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(tf_session, ckpt.model_checkpoint_path)
                    print("[INFO] restored historical model successfully...")
                else:
                    print("[INFO] no any historical model to restore...")

    def classifier(self, tf_session, text):

        with tf_session.as_default():
            with tf_session.graph.as_default():  # 定义属于计算图graph的张量和操作

                    text      = text.strip()
                    seg_list  = list(jieba.cut(text)) # <class 'list'>: ['感冒', 'disease', '吃', '什么', '药']
                    x_data    = self.text_cls_data_loader.text_input_to_array(' '.join(seg_list)) # 找到分词在词典中对应的id，x_data=[[1270 4278 2465 4270 6940    0    0    0    0    0    0    0    0    0   0    0    0    0    0    0]]
                    feed_dict = {self.text_cnn.input_x: x_data,
                                 self.text_cnn.dropout_keep_prob: self.text_cls_dropout_prob}
                    predict   = tf_session.run([self.text_cnn.predictions], feed_dict)

                    return predict[0]


if __name__ == '__main__':
    import settings
    setting = settings.setting()

    graph                = tf.Graph()
    log_device_placement = True  # 是否打印设备分配日志
    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    gpu_options          = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
    session_conf         = tf.compat.v1.ConfigProto(gpu_options=gpu_options,
                                          allow_soft_placement=allow_soft_placement,
                                          log_device_placement=log_device_placement)

    sess     = tf.compat.v1.Session(graph=graph, config=session_conf)
    test_obj = text_classifier(sess, setting)
    while True:
        text = input("[INFO] Please input a chinese sentence below：\n")
        if text == 'exit' or text == 'quit':
            print('[INFO] Bye...')
            break
        predict = test_obj.classifier(sess, text)
        print("[RESULT] %s belongs to class %d ..." % (text, predict))
