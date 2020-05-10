# -*- coding: UTF-8 -*-
'''
Created on Wed May 6 18:33:25 2020

@author: Damon Li
'''

import tensorflow as tf
from text_classification_module import text_classifier
from name_entity_recognition_module import ner
from py2neo import Graph
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加
log_device_placement = True  # 是否打印设备分配日志
allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
gpu_options          = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
session_conf         = tf.ConfigProto(gpu_options=gpu_options,
                                      allow_soft_placement=allow_soft_placement,
                                      log_device_placement=log_device_placement)

class question_analysis(object):

    def __init__(self, setting):
        self.device                      = setting.cpu_id
        self.state_notes_dict            = setting.state_notes_dict #state2entityType
        self.reverse_state_dict          = dict((v, k) for k, v in setting.state_dict.items()) #id2state
        self.question_label_dict         = setting.question_label_dict #label2id
        self.reverse_question_label_dict = dict((v, k) for k, v in self.question_label_dict.items()) #id2label

        self.graph_1 = tf.Graph()
        self.graph_2 = tf.Graph()

        self.ner_session      = tf.Session(graph=self.graph_1, config=session_conf) #指定当前会话所运行的计算图
        self.text_cls_session = tf.Session(graph=self.graph_2, config=session_conf)

        self.ner_obj      = ner(self.ner_session, setting)
        self.text_cls_obj = text_classifier(self.text_cls_session, setting)

        self.host      = setting.neo_host
        self.port      = setting.neo_http_port
        self.user      = setting.neo_user
        self.password  = setting.neo_password
        self.num_limit = setting.neo_search_limit

        self.neo_graph = Graph(host=self.host,
                               http_port=self.port,
                               user=self.user,
                               password=self.password)


    def qa_builder(self, text):
        args_dict          = dict()
        result_dict        = dict()
        question_type_list = list()

        words_x_list, label_list, seq_len_list = self.ner_obj.ner_builder(self.ner_session, text)

        for idx in range(len(words_x_list)):
            middle_question_list = list()
            _entity = ''
            for elem_id in range(seq_len_list[idx]):
                middle_question_list.append(words_x_list[idx][elem_id])
                _entityType = self.reverse_state_dict[int(label_list[idx][elem_id])]
                if _entityType[0] == 'B' or _entityType[0] == 'I':
                    _entity += words_x_list[idx][elem_id]
                elif _entityType[0] == 'E' or _entityType[0] == 'S':
                    _entity += words_x_list[idx][elem_id]
                    _entityType_short = _entityType[-3:]
                    middle_question_list.append(self.state_notes_dict[_entityType_short])
                    if _entity not in args_dict:
                        args_dict.setdefault(_entity, [self.state_notes_dict[_entityType_short]])
                    else:
                        args_dict[_entity].append(self.state_notes_dict[_entityType_short])
                    _entity = ''
                else:
                    _entity = ''
            question_text = ''.join(middle_question_list)
            _classify_idx = self.text_cls_obj.classifier(self.text_cls_session, question_text)
            _classify_label = self.reverse_question_label_dict[_classify_idx[0]]
            question_type_list.append(_classify_label)
        result_dict['args'] = args_dict
        result_dict['question_types'] = question_type_list

        return result_dict


    def build_entity_dict(self, params_dict):
        # 构建实体节点
        entity_dict = dict()
        for param, types in params_dict.items():
            for type in types:
                if type not in entity_dict:
                    entity_dict[type] = [param]
                else:
                    entity_dict[type].append(param)

        return entity_dict


    def question_parser(self, result_dict):
        # 对qa_builder函数返回的result进行解析处理
        args_dict      = result_dict['args']
        entity_dict    = self.build_entity_dict(args_dict)
        question_types = result_dict['question_types']

        cypher_dict_list = list() # each element of list is 'dict' type
        for question_type in question_types:
            _cypher_dict                  = dict()
            _cypher_dict['question_type'] = question_type

            cypher_list = list()
            if question_type == 'disease_symptom':
                cypher_list = self.question_to_cypher(question_type, entity_dict.get('disease'))

            elif question_type == 'symptom_disease' or question_type == ' symptom_curway':
                cypher_list = self.question_to_cypher(question_type, entity_dict.get('symptom'))

            elif question_type == 'disease_drug':
                cypher_list = self.question_to_cypher(question_type, entity_dict.get('disease'))

            elif question_type == 'drug_disease':
                cypher_list = self.question_to_cypher(question_type, entity_dict.get('drug'))

            elif question_type == 'disease_check':
                cypher_list = self.question_to_cypher(question_type, entity_dict.get('disease'))

            elif question_type == 'disease_prevent':
                cypher_list = self.question_to_cypher(question_type, entity_dict.get('disease'))

            elif question_type == 'disease_lasttime':
                cypher_list = self.question_to_cypher(question_type, entity_dict.get('disease'))

            elif question_type == 'disease_cureway':
                cypher_list = self.question_to_cypher(question_type, entity_dict.get('disease'))

            elif question_type == 'disease_desc':
                cypher_list = self.question_to_cypher(question_type, entity_dict.get('disease'))

            if cypher_list:
                _cypher_dict['cypher'] = cypher_list
                cypher_dict_list.append(_cypher_dict)

        return cypher_dict_list


    def question_to_cypher(self, question_type, entities):
        # 针对不同的问题，分开进行处理
        if not entities:
            return None

        # 查询语句
        cypher_list = list()
        # 查询疾病的原因
        if question_type == 'disease_cause':
            cypher_list = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.cause".format(i) for i in entities]

        # 查询疾病的防御措施
        elif question_type == 'disease_prevent':
            cypher_list = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.prevent".format(i) for i in entities]

        # 查询疾病的持续时间
        elif question_type == 'disease_lasttime':
            cypher_list = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.cure_lasttime".format(i) for i in entities]

        # 查询疾病的治愈概率
        elif question_type == 'disease_cureprob':
            cypher_list = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.cured_prob".format(i) for i in entities]

        # 查询疾病的治疗方式
        elif question_type == 'disease_cureway':
            cypher_list = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.cure_way".format(i) for i in entities]

        # 查询疾病的易发人群
        elif question_type == 'disease_easyget':
            cypher_list = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.easy_get".format(i) for i in entities]

        # 查询疾病的相关介绍
        elif question_type == 'disease_desc':
            cypher_list = ["MATCH (m:Disease) where m.name = '{0}' return m.name, m.desc".format(i) for i in entities]

        # 查询疾病有哪些症状
        elif question_type == 'disease_symptom':
            cypher_list = ["MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]

        # 查询症状会导致哪些疾病
        elif question_type == 'symptom_disease':
            cypher_list = ["MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]

        # 查询疾病的并发症
        elif question_type == 'disease_acompany':
            cypher_1 = ["MATCH (m:Disease)-[r:acompany_with]->(n:Disease) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
            cypher_2 = ["MATCH (m:Disease)-[r:acompany_with]->(n:Disease) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
            cypher_list = cypher_1 + cypher_2
        # 查询疾病的忌口
        elif question_type == 'disease_not_food':
            cypher_list = ["MATCH (m:Disease)-[r:no_eat]->(n:Food) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]

        # 查询疾病建议吃的东西
        elif question_type == 'disease_do_food':
            cypher_1 = ["MATCH (m:Disease)-[r:do_eat]->(n:Food) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
            cypher_2 = ["MATCH (m:Disease)-[r:recommand_eat]->(n:Food) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
            cypher_list = cypher_1 + cypher_2

        # 已知忌口查疾病
        elif question_type == 'food_not_disease':
            cypher_list = ["MATCH (m:Disease)-[r:no_eat]->(n:Food) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]

        # 已知推荐查疾病
        elif question_type == 'food_do_disease':
            cypher_1 = ["MATCH (m:Disease)-[r:do_eat]->(n:Food) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
            cypher_2 = ["MATCH (m:Disease)-[r:recommand_eat]->(n:Food) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
            cypher_list = cypher_1 + cypher_2

        # 查询疾病常用药品－药品别名记得扩充
        elif question_type == 'disease_drug':
            cypher_1 = ["MATCH (m:Disease)-[r:common_drug]->(n:Drug) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
            cypher_2 = ["MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
            cypher_list = cypher_1 + cypher_2

        # 已知药品查询能够治疗的疾病
        elif question_type == 'drug_disease':
            cypher_1 = ["MATCH (m:Disease)-[r:common_drug]->(n:Drug) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
            cypher_2 = ["MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]
            cypher_list = cypher_1 + cypher_2

        # 查询疾病应该进行的检查
        elif question_type == 'disease_check':
            cypher_list = ["MATCH (m:Disease)-[r:need_check]->(n:Check) where m.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]

        # 已知检查查询疾病
        elif question_type == 'check_disease':
            cypher_list = ["MATCH (m:Disease)-[r:need_check]->(n:Check) where n.name = '{0}' return m.name, r.name, n.name".format(i) for i in entities]

        return cypher_list


    def cypher_to_answer(self, cypher_dict_list):
        answer_list = list()
        for cypher_dict in cypher_dict_list:
            question_type = cypher_dict['question_type']
            cypher_list   = cypher_dict['cypher']
            _answer_list   = list()
            for query in cypher_list:
                _answer_list += self.neo_graph.run(query).data()
            final_answer = self.answer_template(question_type, _answer_list)
            if final_answer:
                answer_list.append(final_answer)

        return answer_list


    def answer_template(self, question_type, answer_list):
        final_answer = list()
        if not answer_list:
            return ''
        if question_type == 'disease_symptom':
            desc = [i['n.name'] for i in answer_list]
            subject = answer_list[0]['m.name']
            final_answer = '{0}的症状包括：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'symptom_disease':
            desc = [i['m.name'] for i in answer_list]
            subject = answer_list[0]['n.name']
            final_answer = '症状{0}可能染上的疾病有：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_cause':
            desc = [i['m.cause'] for i in answer_list]
            subject = answer_list[0]['m.name']
            final_answer = '{0}可能的成因有：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_prevent':
            desc = [i['m.prevent'] for i in answer_list]
            subject = answer_list[0]['m.name']
            final_answer = '{0}的预防措施包括：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_lasttime':
            desc = [i['m.cure_lasttime'] for i in answer_list]
            subject = answer_list[0]['m.name']
            final_answer = '{0}治疗可能持续的周期为：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_cureway':
            desc = [';'.join(i['m.cure_way']) for i in answer_list]
            subject = answer_list[0]['m.name']
            final_answer = '{0}可以尝试如下治疗：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_cureprob':
            desc = [i['m.cured_prob'] for i in answer_list]
            subject = answer_list[0]['m.name']
            final_answer = '{0}治愈的概率为（仅供参考）：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_easyget':
            desc = [i['m.easy_get'] for i in answer_list]
            subject = answer_list[0]['m.name']

            final_answer = '{0}的易感人群包括：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_desc':
            desc = [i['m.desc'] for i in answer_list]
            subject = answer_list[0]['m.name']
            final_answer = '{0},熟悉一下：{1}'.format(subject,  '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_acompany':
            desc1 = [i['n.name'] for i in answer_list]
            desc2 = [i['m.name'] for i in answer_list]
            subject = answer_list[0]['m.name']
            desc = [i for i in desc1 + desc2 if i != subject]
            final_answer = '{0}的症状包括：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_not_food':
            desc = [i['n.name'] for i in answer_list]
            subject = answer_list[0]['m.name']
            final_answer = '{0}忌食的食物包括有：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_do_food':
            do_desc = [i['n.name'] for i in answer_list if i['r.name'] == '宜吃']
            recommand_desc = [i['n.name'] for i in answer_list if i['r.name'] == '推荐食谱']
            subject = answer_list[0]['m.name']
            final_answer = '{0}宜食的食物包括有：{1}\n推荐食谱包括有：{2}'.format(subject, ';'.join(list(set(do_desc))[:self.num_limit]), ';'.join(list(set(recommand_desc))[:self.num_limit]))

        elif question_type == 'food_not_disease':
            desc = [i['m.name'] for i in answer_list]
            subject = answer_list[0]['n.name']
            final_answer = '患有{0}的人最好不要吃{1}'.format('；'.join(list(set(desc))[:self.num_limit]), subject)

        elif question_type == 'food_do_disease':
            desc = [i['m.name'] for i in answer_list]
            subject = answer_list[0]['n.name']
            final_answer = '患有{0}的人建议多试试{1}'.format('；'.join(list(set(desc))[:self.num_limit]), subject)

        elif question_type == 'disease_drug':
            desc = [i['n.name'] for i in answer_list]
            subject = answer_list[0]['m.name']
            final_answer = '{0}通常的使用的药品包括：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'drug_disease':
            desc = [i['m.name'] for i in answer_list]
            subject = answer_list[0]['n.name']
            final_answer = '{0}主治的疾病有{1},可以试试'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'disease_check':
            desc = [i['n.name'] for i in answer_list]
            subject = answer_list[0]['m.name']
            final_answer = '{0}通常可以通过以下方式检查出来：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        elif question_type == 'check_disease':
            desc = [i['m.name'] for i in answer_list]
            subject = answer_list[0]['n.name']
            final_answer = '通常可以通过{0}检查出来的疾病有{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))

        return final_answer



if __name__ == "__main__":
    import settings
    setting = settings.setting()

    obj_qa = question_analysis(setting)

    while True:
        text = input("[INFO] Please input a chinese sentence below：\n")
        if text == 'exit' or text == 'quit': # eg. 高烧不退怎么办
            print('[INFO] Bye...')
            break
        result_dict = obj_qa.qa_builder(text)
        print('[RESULT] ', result_dict)
        # [RESULT]  {'args': {'高烧': ['symptom'], '不退怎么办': ['disease']}, 'question_types': ['disease_cureway']}

        cypher_dict_list = obj_qa.question_parser(result_dict)
        print('[CYPHERS] ', cypher_dict_list)

        alternative_answer = '[INFO] I can not understand your question, please ask in another way...'
        answer_list = obj_qa.cypher_to_answer(cypher_dict_list)
        if not answer_list:
           print(alternative_answer)
        else:
           print('\n'.join(answer_list))