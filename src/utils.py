# -*- coding: UTF-8 -*-
'''
Created on Wed May 6 16:11:21 2020

@author: Damon Li
'''

from tensorflow.contrib import learn
import numpy as np
import re
import json
import jieba
import copy


class text_cls_data_loader(object):

    def __init__(self, setting):
        self.config = setting
        self.vocab_dir              = setting.text_cls_vocab_dir
        self.num_classes            = setting.num_classes
        self.train_data_dir         = setting.text_cls_train_dir
        self.max_sentence_len       = setting.text_cls_sentence_len
        self.val_sample_ratio       = setting.text_cls_val_sample_ratio
        self.text_cls_embedding_dir = setting.text_cls_embedding_dir

        self.embedding            = None
        self.embedding_shape      = 0

        self.word_id_dict         = dict()
        self.reverse_word_id_dict = dict()


    def load_embedding(self, debug=False):
        self.embedding       = np.load(self.text_cls_embedding_dir)
        self.embedding_shape = np.shape(self.embedding)[-1]

        with open(self.vocab_dir, encoding="utf-8") as json_file:
            self.word_id_dict = json.load(json_file)

        self.reverse_word_id_dict = {}
        for each in self.word_id_dict:  # each 是word_id_dict 字典的key 不是(key，value)组合
            self.reverse_word_id_dict.setdefault(self.word_id_dict[each], each)

        if debug:
            return self.embedding, self.embedding_shape, self.word_id_dict, self.reverse_word_id_dict


    def text_input_to_array(self, text, debug=False):
        text_array = np.zeros([1, self.config.text_cls_sentence_len], dtype=np.int32) # self.config.text_cls_sentence_len=20
        data_line  = text.strip().split(' ')
        for pos in range(min(len(data_line), self.config.text_cls_sentence_len)):
            text_array[0, pos] = int(self.reverse_word_id_dict.get(data_line[pos], 0))

        if debug:
            return text_array, data_line

        return text_array


    def load_raw_data(self, filepath):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        train_datas = []
        with open(filepath, 'r', encoding='utf-8',errors='ignore') as f:
            train_datas = f.readlines()
        one_hot_labels = []
        x_datas = []
        for line in train_datas:
            parts = line.encode('utf-8').decode('utf-8-sig').strip().split(' ',1)
            if len(parts)<2 or (len(parts[1].strip()) == 0):
                continue
            x_datas.append(parts[1])
            one_hot_label = [0]*self.num_classes
            label = int(parts[0])
            one_hot_label[label] = 1
            one_hot_labels.append(one_hot_label)
        print (' data size = ' ,len(train_datas))
        return [x_datas, np.array(one_hot_labels)]


    def load_data(self):
        """Loads starter word-vectors and train/dev/test data."""
        print("Loading word2vec and textdata...")
        x_text, y = self.load_raw_data(self.train_data_path)

        max_document_length = max([len(x.split(" ")) for x in x_text])
        print('len(x) = ', len(x_text), ' ', len(y))
        print(' max_document_length = ', max_document_length)
        x = []
        x = self.get_data_idx(x_text)
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        dev_sample_index = -1 * int(self.dev_sample_percentage * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

        return x_train, x_dev, y_train, y_dev


    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()


    def load_data_and_labels(self, positive_data_file, negative_data_file):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
        negative_examples = [s.strip() for s in negative_examples]
        # Split by words
        x_text = positive_examples + negative_examples
        x_text = [self.clean_str(sent) for sent in x_text]
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)
        return [x_text, y]


    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]


    def data_processing(self, train_data_dir):
        print("[INFO] Loading data...")
        x_text, y = text_cls_data_loader.load_raw_data(train_data_dir)

        # Build vocabulary
        max_document_length = max([len(x.split(" ")) for x in x_text])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x_text)))

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # Split train/test set
        # TODO: This is very crude, should use cross-validation
        dev_sample_index = -1 * int(self.val_sample_ratio * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

        del x, y, x_shuffled, y_shuffled

        print("[INFO] Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("[INFO] Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

        return x_train, y_train, vocab_processor, x_dev, y_dev


class ner_data_loader(object):

    def __init__(self, setting):
        self.config             = setting

        self.train_data_dir     = setting.ner_train_dir
        self.train_label_dir    = setting.ner_train_label_dir
        self.test_data_dir      = setting.ner_test_dir
        self.test_label_dir     = setting.ner_test_label_dir

        self.vocab_dir          = setting.ner_vocab_dir
        self.state_dict         = setting.state_dict
        self.batch_size         = setting.ner_batch_size
        self.sentence_len       = setting.ner_sentence_len
        self.embedding_dir      = setting.ner_embedding_dir
        self.val_sample_ratio   = setting.ner_val_sample_ratio
        self.reverse_state_dict = dict((v, k) for k, v in self.state_dict.items()) # id2state

        self.train_data_raw  = list()
        self.train_label_raw = list()
        self.valid_data_raw  = list()
        self.valid_label_raw = list()
        self.test_data_raw   = list()
        self.test_label_raw  = list()

        self.embedding            = None
        self.embedding_shape      = 0

        self.word_id_dict         = None
        self.reverse_word_id_dict = None


    def load_embedding(self, debug=False):
        self.embedding       = np.load(self.embedding_dir)
        self.embedding_shape = np.shape(self.embedding)[-1]

        with open(self.vocab_dir, encoding="utf8") as file:
            self.word_id_dict = json.load(file)

        self.reverse_word_id_dict = {}
        for each in self.word_id_dict:
            self.reverse_word_id_dict.setdefault(self.word_id_dict[each], each)

        if debug:
            return self.embedding, self.embedding_shape, self.word_id_dict, self.reverse_word_id_dict


    def input_text_process(self, text, debug=False):
        # 初步处理输入文本
        words_x_list           = list()
        seq_len_list           = list()
        output_x_list          = list()
        raw_input_list         = list()
        data_cut_by_jieba_list = list(jieba.cut(text.strip()))

        count = len(data_cut_by_jieba_list) // self.sentence_len

        if len(data_cut_by_jieba_list) % self.sentence_len:
            count += 1

        for j in range(count):
            raw_input_list.append(data_cut_by_jieba_list[j*self.sentence_len: (j+1)*self.sentence_len])

        for idx, raw_input in enumerate(raw_input_list):
            _words = [word for word in raw_input]
            seq_len_list.append(min(self.sentence_len, len(raw_input)))
            _data_trans = [int(self.reverse_word_id_dict.get(word, 0)) for word in raw_input]

            pad_data = self.pad_sequence(_data_trans, self.sentence_len, 0) # 填充
            output_x_list.append(pad_data)
            words_x_list.append(_words)
            if debug:
                print('[DEBUG] ', str(idx), ' _words = ', _words, '\n')
                print('[DEBUG] ', str(idx), ' _data_trans = ', _data_trans, '\n')
                print('[DEBUG] ', str(idx), ' pad_data =', pad_data, '\n')

        if debug:
            return raw_input_list, output_x_list, words_x_list, seq_len_list, count, data_cut_by_jieba_list

        return words_x_list, output_x_list, seq_len_list


    def pad_sequence(self, seq, obj_len, pad_value=None):
        '''
        :param seq: 待填充的序列
        :param obj_len:  填充的目标长度
        :return:
        '''
        seq_copy = copy.deepcopy(seq[:obj_len]) #若seq过长就截断，若短于obj_len就复制全部元素
        if pad_value is None:
            seq_copy = seq_copy * (1 + int((0.5 + obj_len) / (len(seq_copy))))
            seq_copy = seq_copy[:obj_len]
        else:
            seq_copy = seq_copy + [pad_value] * (obj_len - len(seq_copy))
        return seq_copy



if __name__ == '__main__':
    import settings

    setting = settings.setting()

    obj = text_cls_data_loader(setting)
    obj.load_embedding()
    text_array = obj.text_input_to_array('你好，我是李自然')
    print(obj.reverse_word_id_dict.get('你好', 0))
    print(obj.reverse_word_id_dict.get('痔宁片', 0))
    print(obj.reverse_word_id_dict.get('儿泻', 0))