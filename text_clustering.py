 # -*- coding=utf-8 -*-
"""
如果你没有安装jieba和gensim两个工具包
请用pip install jieba,gensim安装
或者下载源代码手动安装（python setup.py install）
"""
import jieba
from gensim import corpora, models
import numpy as np
from sklearn import metrics
from scipy.sparse import *
from scipy import *
import sys
reload(sys)
sys.setdefaultencoding('utf8')
def is_cn_char(qword):
    return ord(qword[0]) >127
def get_cnstr(ori_str):
    cn_str = ''
    for c in ori_str:
        if is_cn_char(c):
            cn_str = cn_str + c
    return cn_str
def liststr(ourlist, sep_str= ' '):
    ourstr = ''
    for item in ourlist: ourstr = ourstr+sep_str+str(item)
    if len(ourstr)>1:return ourstr[1:]
    else: return ourstr

def tokenize(sentence):
    cn_sent = get_cnstr(sentence)
    term_list = jieba.lcut(cn_sent, cut_all=False)
    final_term_list = [term for term in term_list if len(term)>1 and is_cn_char(term)]
    return final_term_list

print 'Word Segmentation...'
segged_file = open('segged_weibo_corpus.txt','w')
for line in open('weibo_corpus.txt'):
    line = line.strip('\n')
    if len(line)==0: continue
    seg_list = tokenize(line)
    term_list = [term for term in seg_list]
    segged_file.write(liststr(term_list,'\t')+'\n')
segged_file.close()

print 'Building the Dictionary...'
segged_corpus = []
fre_t = 2
for line in open('segged_weibo_corpus.txt'):
    line=line.strip('\n')
    segged_corpus.append(line.split('\t'))
dictionary = corpora.Dictionary(segged_corpus)
remove_fre_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq < fre_t]
dictionary.filter_tokens(remove_fre_ids)
dictionary.compactify() 
dictionary.save_as_text('corpus_dictionary.txt')   

print 'Generating BOW Vectors and corresponding Sparse Matrix...'
row_i_indexs=[]
col_t_indexs =[]
value_t_value=[]  
n_instance = 0  
bow_file = open('bow_weibo_corpus.txt','w')  
for segged_item in segged_corpus:
    doc_bow = dictionary.doc2bow(segged_item)
    bow_file.write(liststr(doc_bow)+'\n')                 
    for [term_id, term_value] in doc_bow:
        row_i_indexs.append(n_instance)
        col_t_indexs.append(term_id)
        value_t_value.append(term_value)
    n_instance = n_instance + 1
    row_i_indexs_array = asarray(row_i_indexs)  
    col_t_indexs_array = asarray(col_t_indexs)
    value_t_value_array = asarray(value_t_value)
sparse_matrix = csr_matrix((value_t_value_array,(row_i_indexs_array,col_t_indexs_array)))   
print 'shape of the sparse matrix', sparse_matrix.shape
bow_file.close()

print 'Clustering...'
from sklearn.cluster import KMeans
num_clusters = 1500
km = KMeans(n_clusters=num_clusters,n_init=10)
km.fit(sparse_matrix)
clusters = km.labels_.tolist()

print 'Result Visualization...'
did2cont_dic = {}
did = 0
for line in open('weibo_corpus.txt'):
    did2cont_dic[did] = line.strip('\n')
    did = did+1
cid2dids_dic = {}
for did in range(0, len(clusters)):
    cid = clusters[did]  
    if not cid2dids_dic.has_key(cid): cid2dids_dic[cid] = []
    cid2dids_dic[cid].append(did)
result_file = open('cluster_weibos.txt','w')
for cid in cid2dids_dic:
    result_file.write('-----------'+str(cid)+'-----------\n')
    for did in cid2dids_dic[cid]:
        result_file.write(str(cid)+'\t'+str(did)+'\t'+did2cont_dic[did] + '\n')
    result_file.write('-----------------------\n')
result_file.close()




