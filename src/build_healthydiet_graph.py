# -*- coding: UTF-8 -*-
'''
Created on Tue May 12 16:22:57 2020

@author: Damon Li
'''

import pymongo
import os

'''
关于食材
q1：食材A是什么（附加额外链接信息）
q2：食材A有什么功效
q3：食材A有什么营养价值
q4：食材A适合什么人群吃
q5：食材A不适合什么人群吃
q6：购买食材A需要注意什么
q7：食材A中具体的营养成分及其含量(每100克)
q8：A是属于食材还是药材
q9：我有症状A（或描述体质），能否吃食材A
q10：食材A能和食材B（或药材B）一起吃吗
q11：食材A可以早上（晚上）吃吗
q12：食材A的热量（卡路里/脂肪）含量高吗
q13：推荐一些具有某些功效（美容抗衰老等）的食材

关于药材
q1：药材A的适用哪些人群（症状）
q2：临床上如何使用药材A
q3：药材A的化学成分有哪些
q4：药材A不适合哪些人群（症状）使用
q5：药材A的性味归经
q6：药材A（异名）是什么（回复正名）
q6：A是药材还是食材
q7：我有症状A（或描述体质），能否吃药材A
q8：药材A能和药材B（或食材B）一起吃吗
q9：药材A可以早上（晚上）吃吗

关于菜肴（食物）
q1：菜肴A是怎么做的（即烹饪步骤）
q2：菜肴A中的食材有哪些（包括主料和辅料）
q3：菜肴A需要放食材B（或辅料）吗
q4：菜肴A做起来复杂（简单）吗
q5：烹饪菜肴A大概需要多长时间
q6：早餐有什么推荐
q7：推荐一些清淡的菜肴
q8：晚餐有什么推荐
q9：推荐一些不含食材A（或辅料A）的菜肴
q10：我刚病好，推荐一些菜肴
q11：这道菜肴大概含有多少热量（或卡路里/脂肪）
q12：推荐一些食材A（或同时问多个食材）的菜肴
q13：晚上想吃一些低卡路里的菜肴（食物），推荐一下
'''


class HeathyDietGraph(object):

    def __init__(self, proj_root_dir='./', host='localhost', port=27017, db_name='', col_name=''):
        conn = pymongo.MongoClient()

        zhongyoo_db = conn['zhongyoo']
        meishichina_db = conn['meishichina']

        herbs_col = conn['herbs']
        recipes_col = meishichina_db['recipes']
        ingredients_col = meishichina_db['ingredients']

        nums = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                     'u', 'v', 'w', 'x', 'y', 'z']
        chinese_stop_words = [i.strip() for i in open(os.path.join(proj_root_dir, 'chinese_stop_words.txt'))]
        self.stop_words = chinese_stop_words + alphabets + nums

        self.key_dict = {'食材名称': 'ingredient_name',
                         '食材简介': 'basic_intros',
                         '食材营养价值': 'nutritive_value',
                         '食材功效': 'consumption_effects',
                         '食材适用人群': 'suitable_crowd',
                         '食材不适用人群': 'unsuitable_crowd',
                         '食材购买技巧': 'shopping_tips',
                         '食材营养成分表': 'nutrition_table',
                         '食材更多信息': 'extra_info_urls',
                         '药材名称': 'herb_name',
                         '药材异名': 'alias_name',
                         '药材性味归经': 'taste_tropism',
                         '药材作用': 'herb_effects',
                         '药材临床应用': 'clinical_application',
                         '药材化学成分': 'chemical_component',
                         '药材使用禁忌': 'herb_taboo',
                         '菜肴名称': 'cuisine_name',
                         '菜肴烹饪小技巧': 'cuisine_little_tips',
                         '菜肴分类': 'cuisine_classes',
                         '菜肴主料': 'cuisine_main_ingredients',
                         '菜肴辅料': 'cuisine_auxiliary_ingredients',
                         '菜肴调味品': 'cuisine_spices',
                         '菜肴烹饪级别': 'cuisine_cooking_level',
                         '菜肴口味': 'cuisine_flavor',
                         '菜肴烹饪耗时': 'cuisine_time_consuming',
                         '菜肴烹饪步骤': 'cuisine_cooking_process'
                         }

        ingredients = list()
        herbs       = list()
        cuisines    = list()
        diseases    = list() # 暂时没有
        symptoms    = list() # 暂时没有

        # 构建节点实体关系

