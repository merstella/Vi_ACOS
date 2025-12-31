#coding=utf-8

import codecs as cs
import os
import sys

base_dir = sys.argv[1]
domian_type = sys.argv[2]
data_dir = sys.argv[3] if len(sys.argv) > 3 else None

cur_dir = base_dir+'/output/Extract-Classify-QUAD/'+domian_type

if not os.path.exists(cur_dir+'_1st'):
    os.makedirs(cur_dir+'_1st')

f = cs.open(cur_dir+'_1st'+'/pred4pipeline.txt', 'r').readlines()
if data_dir is None:
    candidates = [
        os.path.join(base_dir, 'Extract-Classify-ACOS', 'tokenized_data'),
        os.path.join(base_dir, 'ACOS-main', 'Extract-Classify-ACOS', 'tokenized_data'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenized_data'),
        os.path.join(base_dir, 'tokenized_data'),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            data_dir = cand
            break
if data_dir is None:
    data_dir = os.path.join(base_dir, 'tokenized_data')

wf = cs.open(os.path.join(data_dir, domian_type+'_test_pair_1st.tsv'), 'w')

for line in f:
    asp = []; opi = []
    line = line.strip().split('\t')
    if len(line) <= 1:
        continue
    text = line[0]
    af = 0
    of = 0
    for ele in line[1:]:
        if ele.startswith('a'):
            asp.append(ele[2:])
            af = 1
        else:
            opi.append(ele[2:])
            of = 1
    if af == 0:
        asp.append('-1,-1')
    if of == 0:
        opi.append('-1,-1')
    if len(asp)>0 and len(opi)>0:
        pred = []

        for pa in asp:
            ast, aed = int(pa.split(',')[0]), int(pa.split(',')[1])
            for po in opi:
                ost, oed = int(po.split(',')[0]), int(po.split(',')[1])
                pred.append([pa, po])
        for ele in pred:  
            wf.write(text+'####'+ele[0]+' '+ele[1]+'\n')
