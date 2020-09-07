# -*- coding: UTF-8 -*-
'''
Created on Thu May 7 16:36:47 2020

@author: Damon Li
'''

from question_classification_module import question_analysis
import settings

setting = settings.setting()

obj_qa = question_analysis(setting)

print('\n*************** Medmart ***************')
# print('[Medmart]: Hello, I am Medmart, nice to meet you! You can enter "exit" or "quit" to turn off the program.')
print('[Medmart]: 您好，我是医学问答机器人Medmart, 希望可以帮到您！您可以输入"exit"或"quit"退出程序。')
while True:
    # alternative_answer = '[Medmart]: I can not understand your question, please ask in another way.'
    alternative_answer = '[Medmart]: 我没能理解您的问题，请换一种方式提问，谢谢！'

    text = input("[User]: ")
    if text == 'exit' or text == 'quit':  # eg. 高烧不退怎么办
        # print('[Medmart]: Good Bye!')
        print('[Medmart]: 再见！')
        break

    result_dict = obj_qa.qa_builder(text) #<class 'dict'>: {'args': {'感冒': ['disease']}, 'question_types': ['disease_drug']}
    if not result_dict:
        print(alternative_answer)
    else:
        print('[RESULT] ', result_dict)
        # [RESULT]  {'args': {'高烧': ['symptom'], '不退怎么办': ['disease']}, 'question_types': ['disease_cureway']}

        cypher_dict_list = obj_qa.question_parser(result_dict) #<class 'list'>: [{'question_type': 'disease_drug', 'cypher': ["MATCH (m:Disease)-[r:common_drug]->(n:Drug) where m.name = '感冒' return m.name, r.name, n.name", "MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) where m.name = '感冒' return m.name, r.name, n.name"]}]
        print('[CYPHERS] ', cypher_dict_list)
        # [CYPHERS]  [{'question_type': 'disease_symptom', 'cypher': ["MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where m.name = '风寒感冒' return m.name, r.name, n.name"]}]

        answer_list = obj_qa.cypher_to_answer(cypher_dict_list)
        # answer_list = <class 'list'>: ['感冒通常的使用的药品包括：洛索洛芬钠胶囊；酚咖片；蒲公英颗粒；肺宁片；风油精；洛索洛芬钠片；
        #                                 依托红霉素片；麻黄止嗽丸；愈美胶囊；消炎片；抗病毒口服液；利巴韦林颗粒；头孢拉定胶囊；感冒灵颗粒；
        #                                 银芩胶囊；头孢丙烯分散片；伤风停胶囊；喉痛灵片；穿心莲片；匹多莫德分散片']
        if not answer_list:
            print(alternative_answer)
        else:
            print('[Medmart]: ', '\n'.join(answer_list))