from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
from collections import OrderedDict
from collections import Counter
import copy
import math

def cosine_sim(vec1,vec2):
    vec1 = [val for val in vec1.values()]#提取值，变成纯向量，之前是字典形式
    vec2 = [val for val in vec2.values()]

    dot_prod = 0
    for i ,v in enumerate(vec1):
        dot_prod += v * vec2[i]

    mag_1 = math.sqrt(sum([x ** 2 for x in vec1]))
    mag_2 = math.sqrt(sum([x ** 2 for x in vec2]))

    return dot_prod/(mag_1 * mag_2)


#文档123组成语料库，建立词汇表
docs = ["The faster Harry got to the store,the faster and faster Harry would get home."]
docs.append("Harry is hairy and faster than Jill.")
docs.append("Jill is not as hairy as Harry.")
print(docs)#17,8,8
doc_tokens =[]
for doc in docs:
    doc_tokens +=[sorted(tokenizer.tokenize(doc.lower()))]#分词，大小写转化,未去重
print(len(doc_tokens[0]))
all_doc_tokens = sum(doc_tokens,[])
print(len(all_doc_tokens))
lexicon = sorted(set(all_doc_tokens))#去重得到词库词汇表 18维
print(len(lexicon))
print(lexicon)


#构建向量模板，词库零向量，确保后面的向量维度相同，将内容填入其中，没有的用零代替
zero_vector = OrderedDict((token,0) for token in lexicon)
print(zero_vector)


#每篇文档的向量表示
doc_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())#分词，大小写归一
    token_counts = Counter(tokens)#生成词袋，去重，计数,key是词，值是数量
    for key,value in token_counts.items():
        vec[key] = value/len(lexicon)
    doc_vectors.append(vec)#每篇文档的tf向量
# print(doc_vectors[0])#用TF值填充
# print(doc_vectors[1])
# print(doc_vectors[2])


document_tfidf_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())  # 分词，大小写归一
    token_counts = Counter(tokens)  # 生成词袋，去重，计数,key是词，值是数量
    for key,value in token_counts.items():#为填充到零向量中做准备#计算每个词在相应文档中的tfidf值
        docs_containing_key = 0
        for _docs in docs:#查询每个文档中是否出现了该词
            if key in _docs:
                docs_containing_key += 1
        tf = value/len(lexicon)
        if docs_containing_key:
            idf = len(docs)/docs_containing_key
        else :
            idf = 0
        vec[key] = tf *idf
    document_tfidf_vectors.append(vec)#每篇文档的tfidf向量
# print(document_tfidf_vectors[0])#用tfidf值填充,获得每篇文档的k维向量
# print(document_tfidf_vectors[1])
# print(document_tfidf_vectors[2])


#小搜索引擎
query = "How long dose it take to get to the store?"
query_vec = copy.copy(zero_vector)
#生成问题的tfidf向量，18维
tokens = tokenizer.tokenize(query.lower())  # 分词，大小写归一
token_counts = Counter(tokens)  # 生成词袋，去重，计数,key是词，值是数量
for key, value in token_counts.items():
    docs_containing_key = 0
    for _docs in docs:
        if key in _docs:
            docs_containing_key += 1
    if docs_containing_key == 0:
            continue
    tf = value / len(lexicon)
    idf = len(docs) / docs_containing_key
    query_vec[key] = tf * idf
print(cosine_sim(query_vec,document_tfidf_vectors[0]))#计算余弦相似度，相似度最高的是答案
print(cosine_sim(query_vec,document_tfidf_vectors[1]))
print(cosine_sim(query_vec,document_tfidf_vectors[2]))


#值最大的，最可能是答案，与问题最相关






