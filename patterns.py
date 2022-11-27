import dataclasses
import random
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd

_DIF = 0


def check_doubles(target):
    target = deepcopy(target)
    for i in range(len(target)):
        target[i] = target[i].replace('-', '')
    return len(set(target)) == len(target)


def con_columns(df, target) -> pd.Series:
    """
    :param df:
    :param target:
    :return: Возвращает серию конъюнкций между значениями target внутри dataframe
    """
    if len(target) == 1:
        return df[target[0]]
    res = df[target[0]] & df[target[1]]
    for i in range(2, len(target)):
        res = res & df[target[i]]
    return res


def random_patterns(data: pd.DataFrame, parts_count=2, iteration=2000, non_neg=True):
    names = list(data.columns)
    names.remove('index_col')
    names.remove('classes_col')
    data_pos = data[data['classes_col'] == 1]
    data_neg = data[data['classes_col'] == 0]
    pos_count = data_pos['classes_col'].count()
    neg_count = data_neg['classes_col'].count()
    patterns = {}
    for _ in range(iteration):
        chosen: List[str] = random.choices(names, k=parts_count)
        chosen.sort()
        if check_doubles(chosen) and '&'.join(chosen) in patterns:
            continue
        if non_neg:
            if (con_columns(data_neg, chosen) == 0).all():
                if (part_range := data_pos[data_pos['classes_col'] == (con_columns(data_pos, chosen))][
                                      'classes_col'].count() / pos_count) != 0:
                    patterns.update({'&'.join(chosen): {'part': chosen, 'range': part_range}})
        else:
            if (con_columns(data_pos, chosen) == 1).all():
                if (part_range := data_neg[data_neg['classes_col'] == (con_columns(data_neg, chosen))][
                                      'classes_col'].count() / neg_count) != 1:
                    patterns.update({'&'.join(chosen): {'part': chosen, 'range': part_range}})
    return patterns


def of_top(data: pd.DataFrame):
    data_pos = data[data['classes_col'] == 1]
    data_neg = data[data['classes_col'] == 0]
    temp = data_pos.copy()
    pos_count = data_pos['classes_col'].count()
    patterns = {}
    while temp['classes_col'].count() != 0:
        row = temp.iloc[[0]]
        row = row.drop(columns=['index_col', 'classes_col'])
        vals = list(row.values[0])

        cols = list(row.columns)
        i = 0
        while i != len(vals):
            if not vals[i]:
                vals.pop(i)
                cols.pop(i)
                i -= 1
            i += 1
        acc = data_pos[data_pos['classes_col'] == (con_columns(data_pos, cols))]['classes_col'].count() / pos_count
        b_acc = 0
        while b_acc == acc:
            cols.sort()
            b_acc = acc
            r_a = ''
            for i in cols:
                cols.remove(i)
                n_acc = data_pos[data_pos['classes_col'] == (con_columns(data_pos, cols))][
                            'classes_col'].count() / pos_count
                if n_acc > b_acc and (con_columns(data_neg, cols) == 0).all():
                    r_a = i
                    b_acc = n_acc
                cols.append(i)
                cols.sort()
            if b_acc != acc:
                cols.remove(r_a)
                acc = b_acc

        patterns.update({'&'.join(cols): {'part': cols, 'range': acc}})
        temp = temp[temp['index_col'] != temp[temp['classes_col'] == (con_columns(data_pos, cols))]['index_col']]
    return patterns


class Pattern:
    def __init__(self, parts, range, error):
        self.error = error
        self.range = range
        self.part = parts

    def __str__(self):
        return '&'.join(self.part)

    def __lt__(self, other):
        return self.error < other.error

    def to_dict(self):
        return {str(self): {
            'part':  self.part,
            'range': self.range,
            'error': self.error
        }}


class Rule:
    def __init__(self, patterns, d_range=0, ga=False, error=0):
        self.ga = ga
        self.error = error
        self.d_range = d_range
        self.patterns = [patterns]

    def compute(self, data):
        con = []
        for i in self.patterns:
            con.append(con_columns(data, i))
        res = pd.Series([0 for i in range(data['classes_col'].count())])
        for i in con:
            res = res | i
        return res

    def __str__(self):
        return ' | '.join([' & '.join(i) for i in self.patterns])

    def com_range(self, data):
        self.d_range = data[data['classes_col'] == self.compute(data)]['classes_col'].count() / data[
            'classes_col'].count()
        return self.d_range

    def clear(self, data):
        if not self.ga:
            self.patterns.sort(reverse=True)
        else:
            sorting = [Pattern() for i in self.patterns]

        check = True
        while check:
            acc = self.com_range(data)
            check = False
            for i in range(len(self.patterns)):
                temp = self.patterns.pop(i)
                if acc == self.com_range(data):
                    check = True
                    break
                else:
                    self.patterns.insert(i, temp)

        self.patterns.sort()


def make_rule(data: pd.DataFrame, patterns):
    rule = Rule(patterns[list(patterns.keys())[0]]['part'])
    for i in patterns:
        if patterns[i]['part'] in rule.patterns or patterns[i]['range'] == 0:
            continue
        if (rule.compute(data) == data['classes_col']).all():
            return rule
        rule.patterns.append(patterns[i]['part'])
    return rule


def ga_chose(names, indv):
    chosen = []
    for i in range(len(indv)):
        if indv[i]:
            if names[i].replace('-', '') in chosen:
                chosen.remove(names[i].replace('-', ''))
                continue
            chosen.append(names[i])
    return chosen


def ga_fitness(data_pos, data_neg, names, indv):
    chosen = ga_chose(names, indv)
    if not chosen:
        return -4
    spray = con_columns(data_pos, chosen)
    spray = spray[spray == 1].count() / data_pos['classes_col'].count()
    error = con_columns(data_neg, chosen)
    error = error[error == 1].count() / data_neg['classes_col'].count()
    return np.sqrt(spray) - np.sqrt(error) * 1.2


class Individ:
    """
    Описание класса, переменных и перегрузок в нем
    """

    def __init__(self, gen: list):
        self.gen = gen
        self.fit = None
        self.length = len(gen)

    def __add__(self, other):
        """
        Перегрузка плюсика
        скрещевание для создания потомков
        """

        model = [random.random() for _ in range(self.length)]
        son = []
        daughter = []
        for i in range(self.length):
            if model[i] < 0.5:
                son.append(self[i])
                daughter.append(other[i])

            else:
                son.append(other[i])
                daughter.append(self[i])

            if i >= _DIF:
                if son[i] == son[i - _DIF] == 1:
                    son[i] = 0
                if daughter[i] == daughter[i - _DIF] == 1:
                    daughter[i] = 0

        return [Individ(son), Individ(daughter)]

    def mutation(self):
        temp = self.gen
        for i in range(self.length):
            if random.random() < 0.15:
                if i >= _DIF and temp[i] == temp[i - _DIF] == 1:
                    continue
                elif i < _DIF and temp[i] == temp[i + _DIF] == 1:
                    continue
                temp[i] = 0 if temp[i] else 1
        self.gen = temp

    def __getitem__(self, item):
        """
        Работа со списком генотипа
        """
        return self.gen[item]

    def __lt__(self, other):
        """
        Перегрузка знака меньше
        """
        if self.fit < other.fit:
            return True
        return False


def genetic_patterns(data: pd.DataFrame, iter_limit=1000):
    data_pos = data[data['classes_col'] == 1]
    data_neg = data[data['classes_col'] == 0]
    pos_count = data_pos['classes_col'].count()
    neg_count = data_neg['classes_col'].count()
    names = list(data.columns)
    names.remove('index_col')
    names.remove('classes_col')

    patterns = {}

    # noinspection PyTypeChecker
    population: list = np.random.randint(0, 2, [100, len(names)]).tolist()  # ignore
    population = [Individ(i) for i in population]
    global _DIF
    _DIF = names.index('-x0^0') - names.index('x0^0')
    for i in population:
        for j in range(i.length // 2):
            if i[j] == i[j + _DIF] == 1:
                if random.random() < 0.5:
                    i.gen[j] = 0
                else:
                    i.gen[j + _DIF] = 0

    for _ in range(iter_limit):
        for i in population:
            i.fit = ga_fitness(data_pos, data_neg, names, i.gen)
        population.sort(reverse=True)
        population = population[:50]
        for i in range(1, len(population)):
            population.extend(population[i - 1] + population[i])

        for i in population:
            i.mutation()

        # if (temp := 100 - len(population)) > 0:  #     population.extend([Individ(bin(i).replace('0b', '').zfill(len(names))) for i in list(
        #         np.random.randint(0, 2 ** len(names) - 1, temp, np.int64))])
    for i in population:
        i.fit = ga_fitness(data_pos, data_neg, names, i.gen)
    population.sort(reverse=True)

    for i in population[:50]:
        temp = ga_chose(names, i.gen)

        spray = con_columns(data_pos, temp)
        spray = spray[spray == 1].count() / data_pos['classes_col'].count()
        error = con_columns(data_neg, temp)
        error = error[error == 1].count() / data_neg['classes_col'].count()
        patterns.update({'&'.join(temp): {'part': temp, 'range': spray, 'fit': i.fit, 'error': error}})
    return patterns
