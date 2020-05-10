# -*- coding: UTF-8 -*-
'''
Created on Wed May 6 18:42:40 2020

@author: Damon Li
'''

import tensorflow as tf
from bilstm_crf_module import BiLSTM_CRF
from utils import ner_data_loader
import logging

class ner(object):

    def __init__(self, tf_session, setting):

        with tf_session.as_default():
            with tf_session.graph.as_default(): # 定义属于计算图graph的张量和操作
                self.ner_dropout_prob = setting.ner_dropout_prob

                self.ner_data_loader = ner_data_loader(setting)
                self.ner_data_loader.load_embedding()

                self.ner_model = BiLSTM_CRF(batch_size=setting.ner_batch_size,
                                            tag_nums=setting.ner_tag_nums,
                                            hidden_nums=setting.ner_hidden_nums,
                                            sentence_len=setting.ner_sentence_len,
                                            word_embeddings=self.ner_data_loader.embedding,
                                            device=setting.gpu_id)

                self.saver = tf.train.Saver(max_to_keep=1)
                ckpt       = tf.train.get_checkpoint_state(setting.ner_model_checkpoint)
                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(tf_session, ckpt.model_checkpoint_path)
                    logging.info('[INFO] loaded ner model successful...')

    def ner_builder(self, tf_session, text):

        with tf_session.as_default():
            with tf_session.graph.as_default(): # 定义属于计算图graph的张量和操作
                if text == ' ':
                    print('[INFO] input text is NULL, ERROR!')
                    return

                words_x_list, output_x_list, seq_len_list = self.ner_data_loader.input_text_process(text)

                feed_dict       = {self.ner_model.input_x: output_x_list,
                                   self.ner_model.sequence_lengths: seq_len_list,
                                   self.ner_model.dropout_keep_prob: self.ner_dropout_prob}
                predicted_label = tf_session.run([self.ner_model.crf_labels], feed_dict) # predicted_label是三维的[1,1,25]，第1维包含了一个矩阵
                label_list      = list()

                for idx in range(len(predicted_label[0])):
                    _label = predicted_label[0][idx].reshape(1, -1)
                    label_list.append(list(_label[0]))

                return words_x_list, label_list, seq_len_list


if __name__ == "__main__":
    import settings
    setting = settings.setting()

    graph                = tf.Graph()
    log_device_placement = True  # 是否打印设备分配日志
    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    gpu_options          = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    session_conf         = tf.ConfigProto(gpu_options=gpu_options,
                                          allow_soft_placement=allow_soft_placement,
                                          log_device_placement=log_device_placement)

    tf_session = tf.Session(graph=graph, config=session_conf)
    test_obj   = ner(tf_session, setting)

    while True:
        text = input("[INFO] Please input a chinese sentence below：\n")
        if text == 'exit' or text == 'quit':
            print('[INFO] Bye...')
            break
        words_x_list, label_list, seq_len_list = test_obj.ner_builder(tf_session, text)
        for idx in range(len(words_x_list)):
            for elem in range(seq_len_list[idx]):
                print('[RESULT] entity -> %s | label id -> %s' % (words_x_list[idx][elem], label_list[idx][elem]))
