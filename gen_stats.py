import argparse
import os
from glob import glob 



import numpy as np
from typing import Optional, Tuple, List, Dict
from torch.utils.data import DataLoader
from collections import defaultdict




parser = argparse.ArgumentParser(description='')


######################### Path Setting #########################
parser.add_argument('--base-dir', type=str,default=None,
                    help='The base dir of this project.')
parser.add_argument('--matched-regex', type=str, default=None)
parser.add_argument('--no-match-regex', type=str, default=None)

parser.add_argument('--metric', type=str, default='accuracy')

parser.add_argument('--save', action='store_true', default=False)

args = parser.parse_args()
    
dir_reg = os.path.join(args.base_dir, args.matched_regex)
candidate_dirs = glob(dir_reg)
if args.no_match_regex is not None:
    no_dir_reg = os.path.join(args.base_dir, args.no_match_regex)
    no_candidate_dirs = glob(no_dir_reg)
    candidate_dirs = [d for d in candidate_dirs if d not in no_candidate_dirs]
print(f"Found {len(candidate_dirs)} directories.")

def parse_file_name_get_ab(file_name: str) -> Tuple[str, str]:
    task_name = file_name.split('/')[-1]
    return task_name.split('-')[1], task_name.split('-')[2]

def get_last_metric_from_log(log_file_path: str, metric='accuracy') -> float:
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
        return float(lines[-2].strip())


ab2metric = defaultdict(dict)
for dir_name in candidate_dirs:
    # extract file matching with reg: log*log
    log_file_path = glob(os.path.join(dir_name, 'log*log'))[0]
    a, b = parse_file_name_get_ab(dir_name)
    m = get_last_metric_from_log(log_file_path, args.metric)
    ab2metric[a][b] = m

sep = '\t'
# print out the metrics in a copiable format to google sheet
a_range = sorted(ab2metric.keys())
b_range = sorted(set(b for a in ab2metric for b in ab2metric[a]))
print('a_range:', a_range)
print('b_range:', b_range)
for b in b_range:
    print(sep, b, end='')
print()
for a in a_range:
    print(round(1-float(a), 2), end=sep)
    for b in b_range:
        if b in ab2metric[a] and ab2metric[a][b] is not None:
            # print out 4 digit precision
            print(round(ab2metric[a][b], 4), end=sep)
        else:
            # print out empty cell
            print('', end=sep)
    print()

if args.save:
    with open(f'{args.base_dir}_{args.metric}.csv', 'w') as f:
        f.write('a_range')
        for b in b_range:
            f.write(sep + b)
        f.write('\n')
        for a in a_range:
            f.write(str(round(1-float(a), 2)))
            for b in b_range:
                if b in ab2metric[a] and ab2metric[a][b] is not None:
                    f.write(sep + str(round(ab2metric[a][b], 4)))
            f.write('\n')


# usage: python gen_stats.py --base-dir ./data-model/cifar100/forgetting --matched-regex budget* --save