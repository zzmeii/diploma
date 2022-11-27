import datetime
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from patterns import random_patterns, make_rule, of_top, genetic_patterns, Pattern, con_columns
from tiny_gp_plus import symbol_regression


class Binarized:
    def __init__(self, data: pd.DataFrame, ):
        self.data: pd.DataFrame = data
        self.binary_tabels = {}
        self.names_dict = dict(zip(list(data.columns), ['index_col', 'classes_col',
                                                        *[f'x{i}' for i in range(len(list(data.columns)) - 2)]]))
        self.data = self.data.rename(columns=self.names_dict)
        self.cut_points = {}

    def one_cut_point(self):
        temp = self.data
        cols = list(self.data.columns)
        cols.remove('index_col')
        cols.remove('classes_col')
        cut_point = {}
        for i in cols:
            cut_point.update({i: [np.average(temp[i])]})
        self.cut_points.update({'OCP': cut_point})

    def three_cp(self):
        temp = self.data
        cols = list(self.data.columns)
        cols.remove('index_col')
        cols.remove('classes_col')
        cut_point = {}
        for i in cols:
            fcp = np.average(temp[i])
            cut_point.update({i: [np.average(temp[temp[i] < fcp][i]), fcp, np.average(temp[temp[i] > fcp][i])]})

        self.cut_points.update({'MCP': cut_point})

    def make_master_table(self, name):

        cols = list(self.data.columns)
        cols.remove('index_col')
        cols.remove('classes_col')
        res = {'index_col': self.data['index_col'], 'classes_col': self.data['classes_col']}
        new_cols = []

        for i in cols:
            counter = 0
            for j in self.cut_points[name][i]:
                temp = self.data
                temp.loc[temp[i] >= j, i] = True
                temp.loc[temp[i] < j, i] = False

                res.update({f'{i}^{counter}': temp[i].astype(int)})
                counter += 1

        df = pd.DataFrame(res)
        for i in res:
            if i != 'index_col' and i != 'classes_col':
                df[f'-{i}'] = np.abs(df[i] - 1)
        return df


def get_line(y):
    res = []
    active_cols = list(y.columns)
    active_cols.remove('index_col')
    active_cols.remove('classes_col')
    for i in range(y['index_col'].count()):
        res.append("".join([str(y[j][i]) for j in active_cols]) + f'={y["classes_col"][i]}')
    return res


class fit:
    def __init__(self, class_label):
        self.class_label = class_label

    def class_fitness(self, first, second, *args, **kwargs):
        temp = [i[:-1] for i in second]
        temp = [first.compute_tree(i) for i in temp]
        second = [(second[i][-1] - temp[i]) < 0 for i in range(len(temp))]
        a = b = c = d = 0
        for i in range(len(second)):
            for j in range(len(second)):
                if self.class_label[i] == self.class_label[j]:
                    if second[i] == second[j]:
                        a += 1
                    else:
                        c += 1
                else:
                    if second[i] != second[j]:
                        b += 1
                    else:
                        d += 1
        return 1 - (a + b) / (a + b + c + d)


def make_uniq_list(target):
    res = []
    for i in target:
        if i not in res:
            res.append(i)
    return res


def start_sr(binary):
    y = binary.make_master_table("OCP")
    master = get_line(y)
    active_cols = list(y.columns)
    active_cols.remove('index_col')
    active_cols.remove('classes_col')
    prep = y[active_cols]
    prep['classes_col'] = y['classes_col']
    fit_func = fit(y['classes_col'].to_list())
    return symbol_regression(make_uniq_list(prep.values.tolist()), error=fit_func.class_fitness, add_x=False,
                             headers=active_cols, prob_mutation=np.random.uniform(0.25, 0.55),
                             xo_rate=np.random.uniform(0.5, 0.8), min_depth=np.random.randint(2, 5),
                             max_depth=np.random.randint(5, 10), pop_size=np.random.randint(70, 100))


def con_gp(patterns, data):
    con = []
    for i in patterns:
        con.append(con_columns(data, i))
    res = pd.Series([0 for i in range(data['classes_col'].count())])
    for i in con:
        res = res | i
    return res


def test_gp(data, count):
    test_res = {"Количество паттернов":             [],
                "Точность классификации обучающая": [],
                "Решающая функция":                 [],
                "Точность классификации тестовая":  [],
                "Количество поколений":             []}
    pop = [300, 500, 1000]
    for _ in range(count):
        print(_)
        training_data, testing_data = train_test_split(data, test_size=0.3)
        training_data.reset_index(drop=True, inplace=True)
        testing_data.reset_index(drop=True, inplace=True)
        x = Binarized(training_data)
        x.one_cut_point()
        train_table = x.make_master_table("OCP")
        y = Binarized(testing_data)
        y.cut_points = x.cut_points
        testing_table = y.make_master_table("OCP")
        res = {}
        res.update(genetic_patterns(train_table, 300))
        g_patt = []
        for i in res:
            g_patt.append(Pattern(res[i]['part'], res[i]['range'], res[i]['error']))
        g_patt.sort(reverse=True)
        pattern_names = [i.part for i in g_patt]

        data_pos = train_table[train_table['classes_col'] == 1]
        data_neg = train_table[train_table['classes_col'] == 0]
        pos_count = data_pos['classes_col'].count()
        neg_count = data_neg['classes_col'].count()
        spray = con_gp(pattern_names, data_pos)
        spray = spray[spray == 1].count() / pos_count
        error = con_gp(pattern_names, data_neg)
        error = error[error == 1].count() / neg_count

        def get_error(patt):
            er_sam = con_gp(patt, data_neg)
            return er_sam[er_sam == 1].count() / neg_count

        def get_spray(patt):
            er_sam = con_gp(patt, data_pos)
            return er_sam[er_sam == 1].count() / pos_count

        check = True

        while check:
            acc = spray
            er = error
            check = False
            for i in range(len(pattern_names)):
                temp = pattern_names.pop(i)
                new_acc = get_spray(pattern_names)
                new_err = get_error(pattern_names)
                if er >= new_err and (er - new_err) > acc - new_acc:
                    check = True
                    break
                else:
                    pattern_names.insert(i, temp)
        test_res['Точность классификации обучающая'].append(
            (train_table[train_table['classes_col'] == con_gp(pattern_names, train_table)]['classes_col'].count() / (
                    pos_count + neg_count)))
        test_res['Решающая функция'].append("|".join(['&'.join(i) for i in pattern_names]))
        test_res['Точность классификации тестовая'].append(
            (testing_table[testing_table['classes_col'] == con_gp(pattern_names, testing_table)][
                 'classes_col'].count() / testing_table['classes_col'].count()))
        test_res['Количество паттернов'].append(len(pattern_names))
        test_res['Количество поколений'].append(300)

    pd.DataFrame(test_res).to_excel('testing GP300NE.xlsx')


if __name__ == '__main__':
    data = pd.read_csv('wdbc.csv', header=None)
    data = data.reindex()
    data.loc[data[1] == 'M', 1] = 1
    data.loc[data[1] == 'B', 1] = 0
    test_gp(data, 20)
    # training_data, testing_data = train_test_split(data, test_size=0.3)
    # x = Binarized(training_data)
    # x.one_cut_point()
    # # x.three_cp()
    # # z = x.make_master_table('MCP')
    # y = x.make_master_table("OCP")
    # res = {}
    # testing = [4, 7, 10, 20, 30]
    # # res.update(genetic_patterns(y))
    # # rule = of_top(y)
    # test_res = {"Максимальный размер паттерна":       [],
    #             'Время работы':                       [],
    #             "Количество паттернов до очистки":    [],
    #             "Количество паттернов после очистки": [],
    #             "Точность классификации обучающая":   [],
    #             "Решающая функция":                   [],
    #             "Точность классификации тестовая":    []}
    # # res.update(random_patterns(y, 3, iteration=2000))
    # # res.update(random_patterns(y, 4, iteration=2000))
    # # res.update(random_patterns(y, 5, iteration=2000))
    # # rule = make_rule(y, res)
    # # rule.clear(y)
    #
    # for tt in testing:
    #     for _ in range(3):
    #         time_start = datetime.datetime.now()
    #         training_data, testing_data = train_test_split(data, test_size=0.3)
    #         training_data.reset_index(drop=True, inplace=True)
    #         testing_data.reset_index(drop=True, inplace=True)
    #         x = Binarized(training_data)
    #         x.one_cut_point()
    #         y = x.make_master_table("OCP")
    #         test_sample = Binarized(testing_data)
    #         test_sample.cut_points = x.cut_points
    #         test_tablet = test_sample.make_master_table("OCP")
    #         for i in range(3, tt):
    #             res.update(random_patterns(y, i, iteration=2000))
    #         rule = make_rule(y, res)
    #         not_in_list = y[y['classes_col'] != rule.compute(y)]
    #         test_res['Максимальный размер паттерна'].append(tt)
    #         test_res['Количество паттернов до очистки'].append(len(rule.patterns))
    #         test_res['Точность классификации обучающая'].append(rule.com_range(y))
    #         test_res['Точность классификации тестовая'].append(rule.com_range(test_tablet))
    #         print(len(rule.patterns))
    #         print(rule.com_range(y))
    #         rule.clear(y)
    #         print(len(rule.patterns))
    #         print(tt)
    #         test_res['Количество паттернов после очистки'].append(len(rule.patterns))
    #         test_res['Время работы'].append((datetime.datetime.now() - time_start).seconds)
    #         test_res['Решающая функция'].append(str(rule))
    # pd.DataFrame(test_res).to_excel('testing2.xlsx')
