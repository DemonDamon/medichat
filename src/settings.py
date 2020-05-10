import math

class setting:

	def __init__(self):

		self.cpu_id = '/cpu:0'
		self.gpu_id = '/gpu:0'

		self.text_cls_test_dir      = './data/text_classification_dataset/test_data.txt'
		self.text_cls_train_dir     = './data/text_classification_dataset/train_data.txt'
		self.text_cls_vocab_dir     = './models/text_classification_models/id2words.vab'
		self.text_cls_embedding_dir = './models/text_classification_models/word2vec_model.npy'

		self.text_cls_dropout_prob     = 1.0
		self.text_cls_sentence_len     = 20
		self.text_cls_num_checkpoints  = 1
		self.text_cls_val_sample_ratio = 0.2
		self.text_cnn_model_save_dir = './models/text_classification_models/trained_saved_model'

		self.ner_test_dir         = './data/ner_dataset/test_data.txt'
		self.ner_train_dir        = './data/ner_dataset/train_cutword_data.txt'
		self.ner_vocab_dir        = './models/ner_models/document.txt.vab'
		self.ner_embedding_dir    = './models/ner_models/document.txt.ebd.npy'
		self.ner_test_label_dir   = './data/ner_dataset/test_label.txt'
		self.ner_train_label_dir  = './data/ner_dataset/label_cutword_data.txt'
		self.ner_model_checkpoint = './models/ner_models/trained_saved_model'

		self.ner_tag_nums         = 13  # 标签数目
		self.ner_batch_size       = 100
		self.ner_hidden_nums      = 650  # bi-lstm的隐藏层单元数目
		self.ner_sentence_len     = 25 # 句子长度,输入到网络的序列长度
		self.ner_dropout_prob     = 1.0
		self.ner_val_sample_ratio = 0.1

		self.num_classes = 9

		self.embedding_dim = 200
		self.filter_sizes = '2 3 4'
		self.l2_lambda = 0.0
		self.num_filters = 128

		self.state_dict = {'O': 0,
					       'B-dis': 1, 'I-dis': 2, 'E-dis': 3,
					       'B-sym': 4, 'I-sym': 5, 'E-sym': 6,
					       'B-dru': 7, 'I-dru': 8, 'E-dru': 9,
					       'S-dis': 10, 'S-sym': 11, 'S-dru': 12}

		self.state_notes_dict = {'dis': 'disease', 'sym': 'symptom', 'dru': 'drug'}

		self.question_label_dict = {"disease_symptom": 0, "symptom_curway": 1, "symptom_disease": 2,
									"disease_drug": 3, "drug_disease": 4, "disease_check": 5,
									"disease_prevent": 6, "disease_lasttime": 7, "disease_cureway": 8}

		self.neo_host      = 'localhost'
		self.neo_http_port = 7474
		self.neo_user      = 'neo4j'
		self.neo_password  = '123456'
		self.neo_search_limit = 20