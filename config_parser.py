import os
import shutil

from pandas import DataFrame


def parse_config(filename) -> dict:
    file = open(filename, 'r')
    lines = file.read()
    file.close()
    if lines == '':
        return {}
    lines = lines.split('\n')
    result = {}
    for i in lines:
        temp = i.split(':')
        if temp == [""]:
            continue
        result.update({temp[0].strip(): temp[1].strip()})

    return result


def make_results():
    template = {'experiment number': [],
                'pop_size':          [],
                'min_depth':         [],
                'max_depth':         [],
                'generations':       [],
                'tournament_size':   [],
                'xo_rate':           [],
                'prob_mutation':     [],
                'bloat_control':     [],
                'add_x':             [],
                'function':          [],
                'full_error':        []}
    folders = os.listdir(r'results')
    for i in folders:

        config_dict = parse_config('results\\' + i + '\\config.txt')
        if config_dict == {}:
            shutil.rmtree('results\\' + i)
            continue
        for j in config_dict:
            template[j].append(config_dict[j])
        template['experiment number'].append(i)
    template.pop('full_error')
    DataFrame(template).to_excel('results.xlsx')


make_results()
