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

def main():
    parser = argparse.ArgumentParser(description='Calc precision and nDCG')
    parser.add_argument('-o', '--ordered', action='store_true', help='Input is already ordered (or sorted)')

    args = parser.parse_args()

    max_k = 5

    dcg_list = [0 for i in range(max_k)]
    idcg_list = [0 for i in range(max_k)]
    dcg_list[0] = 1.0
    idcg_list[0] = 1.0
    for i in range(1, max_k):
        dcg_list[i] = 1.0 / math.log(i + 2, 2)
        idcg_list[i] = idcg_list[i-1] + dcg_list[i]

    num_lines = 0
    accs = [0 for i in range(max_k)]
    ndcgs = [0.0 for x in range(max_k)]

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

        dcgs = [0.0 for x in range(max_k)]

        for i, p in enumerate(k_list):
            if p in l_set:
                accs[i] += 1
                dcgs[i] = dcg_list[i]

        sum_dcg = 0.0
        for i, dcg in enumerate(dcgs):
            sum_dcg += dcg
            ndcgs[i] += sum_dcg / idcg_list[min(i, len(l_set)-1)]


    print("#samples={0}".format(num_lines))

    a_sum = 0.0
    for n, a in enumerate(accs):
        a_sum += float(a)
        p_at_n = a_sum / num_lines / (n + 1)
        print("P@{0}={1:.6f}".format(n+1, p_at_n))

    for n, ndcg in enumerate(ndcgs):
        print("nDCG@{0}={1:.6f}".format(n+1, ndcg / num_lines))


if __name__ == '__main__':
    main()
