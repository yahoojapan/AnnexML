#
# Copyright (C) 2017 Yahoo Japan Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import sys
import math
import argparse
from collections import defaultdict

def calc_propensity_score(train_file, A, B):
    num_inst = 0
    freqs = defaultdict(int)

    for line in open(train_file):
        if line.find(":") == -1:
            continue # header line

        num_inst += 1

        idx = line.find(" ")
        labels = line[:idx]

        for l in labels.split(","):
            l = int(l)
            freqs[l] += 1

    C = (math.log(num_inst) - 1) * pow(B+1, A)

    pw_dict = dict()
    for k in freqs.keys():
        pw_dict[k] = 1 + C * pow(freqs[k]+B, -A)

    default_pw = 1 + C * pow(B, -A)

    return pw_dict, default_pw



def main():
    parser = argparse.ArgumentParser(description='Calc propensity scored precision and nDCG (PSP@k and PSnDCG@k)')
    parser.add_argument('train_file', help='Input train file for calculating propensity score')
    parser.add_argument('-o', '--ordered', action='store_true', help='Input is already ordered (or sorted)')
    parser.add_argument('-A', '--A', type=float, default=0.55, help='A')
    parser.add_argument('-B', '--B', type=float, default=1.5, help='B')

    args = parser.parse_args()

    max_k = 5

    pw_dict, default_pw = calc_propensity_score(args.train_file, args.A, args.B)

    dcg_list = [0 for i in range(max_k)]
    idcg_list = [0 for i in range(max_k)]
    dcg_list[0] = 1.0
    idcg_list[0] = 1.0
    for i in range(1, max_k):
        dcg_list[i] = 1.0 / math.log(i + 2, 2)
        idcg_list[i] = idcg_list[i-1] + dcg_list[i]

    num_lines = 0
    n_accs = [0 for i in range(max_k)]
    d_accs = [0 for i in range(max_k)]
    n_ndcgs = [0.0 for x in range(max_k)]
    d_ndcgs = [0.0 for x in range(max_k)]

    for line in sys.stdin:
        num_lines += 1
        tokens = line.rstrip().split()
        if len(tokens) != 2:
            continue
        ls = tokens[0]
        ps = tokens[1]

        l_set = set([int(x) for x in ls.split(",")])

        k_list = list()
        pred_dict = dict()
        for t in ps.split(","):
            p, v = t.split(":")
            p = int(p)
            v = float(v)
            pred_dict[p] = v
            k_list.append(p)

        if not args.ordered:
            # compatibility for (old) Matlab scripts
            k_list = sorted([k for k in pred_dict.keys()], key=lambda x: (-pred_dict[x], x))

        if len(k_list) > max_k:
            k_list = k_list[:max_k]

        n_dcgs = [0.0 for x in range(max_k)]
        d_dcgs = [0.0 for x in range(max_k)]

        for i, p in enumerate(k_list):
            pw = pw_dict[p]
            if p in l_set:
                n_accs[i] += pw
                n_dcgs[i] = pw * dcg_list[i]

        sum_n_dcg = 0.0
        for i, n_dcg in enumerate(n_dcgs):
            sum_n_dcg += n_dcg
            n_ndcgs[i] += sum_n_dcg / idcg_list[min(i, len(l_set)-1)]

        
        l_pw_list = list()
        for l in l_set:
            pw = pw_dict[l] if l in pw_dict else default_pw
            l_pw_list.append(pw)
        l_pw_list = sorted(l_pw_list, reverse=True)

        if len(l_pw_list) > max_k:
            l_pw_list = l_pw_list[:max_k]

        for i, pw in enumerate(l_pw_list):
            d_accs[i] += pw
            d_dcgs[i] = pw * dcg_list[i]

        sum_d_dcg = 0.0
        for i, d_dcg in enumerate(d_dcgs):
            sum_d_dcg += d_dcg
            d_ndcgs[i] += sum_d_dcg / idcg_list[min(i, len(l_set)-1)]


    print("#samples={0}".format(num_lines))

    n_a_sum, d_a_sum = 0.0, 0.0
    for n in range(max_k):
        n_a_sum += float(n_accs[n])
        d_a_sum += float(d_accs[n])
        print("PSP@{0}={1:.6f}".format(n+1, n_a_sum / d_a_sum))

    for n in range(max_k):
        print("PSnDCG@{0}={1:.6f}".format(n+1, n_ndcgs[n] / d_ndcgs[n]))


if __name__ == '__main__':
    main()
