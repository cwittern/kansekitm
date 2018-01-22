#! /usr/bin/python
# -*- encoding: utf-8 -*-
# prepare all models in a directory for analysis

import gensim, os, pickle, warnings, mmseg, kansekitm, logging
warnings.simplefilter("ignore", category=DeprecationWarning)
from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity, Similarity
from collections import defaultdict
kstm_base = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
t = mmseg.Dictionary()
t.load_chars('%s/dic/chars.dic' % (kstm_base ))
largedic='%s/dic/words.dic' % (kstm_base )
#t.load_words('%s/dic/words.dic' % (kstm_base ))
t.load_words('%s/dic/frequentwords.dic' % (kstm_base ))
kansekitm.t = t
def tokenizer (s, tok="mmseg"):
    global t
    largedic_loaded = t.has_word(u"自在將軍")
    if tok == "1gram":
        return kansekitm.ngrams(s, 1)
    elif not tok=="largedic" and largedic_loaded:
        t = mmseg.Dictionary()
        t.load_chars('%s/dic/chars.dic' % (kstm_base ))
        t.load_words('%s/dic/frequentwords.dic' % (kstm_base ))
        kansekitm.t = t
    elif tok=="largedic" and not largedic_loaded:
        t.load_words(largedic)
    return kansekitm.mmseg_split(s)

#load data models from saved files
def prepan(m):
    vis=[]
    mp = "data/%s" % m
    fx = ["%s/%s"%(mp, a) for a in os.listdir(mp) if a.endswith(".model")]
    for f in fx:
        base = f[:-6]
        desc = os.path.split(base)[-1].split("-")[1]
        lda = gensim.models.ldamodel.LdaModel.load(f)
        corpus = pickle.load(open("%s.corpus" % (base)))
        texts = pickle.load(open("%s.texts" % (base)))
        dictionary = gensim.corpora.Dictionary.load("%s.dictionary" % (base))
        index = MatrixSimilarity(corpus, num_features=len(dictionary))
        vis.append((lda, corpus, dictionary, texts, index, desc))
    return sorted(vis, key = lambda x : x[-1])

# plot the alpha distribution for all models
def print_alpha(models):
    pass

def topics_for_term(models, term):
    res = []
    for lda, corpus, dictionary, texts, index, desc in models:
        res.append((desc, lda.get_term_topics(dictionary.token2id[term])))
    return res

def list_topics(lda, num_words=10):
    res=[]
    alpha = list(lda.alpha)
    for t in lda.show_topics(num_topics=lda.num_topics, num_words=num_words, formatted=False):
        res.append((t[0], alpha[t[0]], " ".join([a[0] for a in t[1]]) ))
    return res

def print_topics(lda, html=False):
    if html:
        template = "<tr><td>%d</td><td>%4.4f</td><td>%s</td></tr>"
        hr = "<table><hr><td>#</td><td>alpha</td><td>Topic Terms</td></hr>\n"
        ft = "</table>"
    else:
        template = "%d\t%4.4f\t%s\n"
        hr = "#\talpha\tTopic Terms"
        ft = ""
    return hr+"\n".join([template % a for a in sorted(list_topics(lda), key=lambda x: x[1], reverse=True)]) + ft        

def topic_terms_table(models, html=True):
    if html:
        template = "<td>%d</td><td>%4.4f</td><td>%s</td>"
        tr = "<tr>%s<tr>\n"
        out = "<table>"
        ft = "</table>"
    else:
        template = "%d\t%4.4f\t%s\n"
        hr = "#\talpha\tTopic Terms %s\t"
        tr = "%s\n"
        out = ""
        ft = ""
    res = []
    labels = []
    cols = defaultdict(list)
    for lda, corpus, dictionary, texts, index, desc in models:
        lt = sorted(list_topics(lda), key=lambda x: x[1], reverse=True)
        labels.append(desc)
        res.append(lt)
    for r in res:
        for i, c in enumerate(r):
            cols[i].append(c)
    if html:
        out += "<hr>%s</hr>\n" % ("".join(["<td>#</td><td>alpha</td><td>Topic Terms (%s)</td>" % (a) for a in labels]))
    else:
        out += "\n" % ("".join([hr % (a) for a in labels]))
    for i in range(len(cols)):
        out += tr % ( "".join([template % a for a in cols[i]]))
    out += ft
    return out
        
def print_all_topics(models, html=False):
    res=[]
    if html:
        template = "<h3>Model: %s</h3>\n<table>%s</table>\n"
    else:
        template = "Model: %s\t%s\n"
    for lda, corpus, dictionary, texts, index, desc in models:
        tl = print_topics(lda, html)
        res.append(template % (desc, tl))
    return res

# topic term cooccurrences:
def get_tt_co(models, num_words=20):
    co=defaultdict(list)    
    for lda, corpus, dictionary, texts, index, desc in models:
        res = list_topics(lda, num_words=num_words)
        for t in res:
            for tl in t[2].split():
                for i, c1 in enumerate(tl):
                    for c2 in tl[i:]:
                        if c1 != c2:
                            l = [c1, c2]
                            l.sort()
                            co[",".join(l)].append("%s:%s"%(desc, t[0]))
    co = list(co.items())
    co = sorted(co, key = lambda x: len(x[1]), reverse=True)
    return co

def tab_co_list(co):
    cnt_co = defaultdict(int)
    for a in co:
        cnt_co[len(a[1])] += 1
    return (cnt_co.keys(), cnt_co.values())

def get_query_docs(models, query, num_best=5):
    res = []
    query="".join(query)
    for lda, corpus, dictionary, texts, index, desc in models:
        q_tok = tokenizer(query, tok=desc)
        q_bow = dictionary.doc2bow(q_tok)
        print u"%s %s %s" % (desc, query, q_bow)
        logging.info(u"%s %s %s" % (desc, query, q_bow))
        q_lda = lda[q_bow]
        index.num_best = num_best
        r = [(a[0], a[1], " ".join(texts[a[0]])) for a in index[q_lda]]
        res.append(("%s:\nQuery:\n%s" % (desc, " ".join(q_tok)), r))
    return res

# 
def get_simdocs(models, docid, num_best=5):
    res = []
    for lda, corpus, dictionary, texts, index, desc in models:
        q_bow = dictionary.doc2bow(texts[docid])
        q_lda = lda[q_bow]
        index.num_best = num_best
        r = [(a[0], a[1], " ".join(texts[a[0]])) for a in index[q_lda]]
        res.append(("%s: Query Document:\n%s" % (desc, " ".join(texts[docid])), r))
    return res

def print_simdocs(res, html=False):
    out = []
    for r in res:
        if html:
            out.append(r[0])
        else:
            out.append("<p>"+"<p/><p>".join(r[0].split("\n")))
        out.append("Response")
        for rq in r[1]:
            out.append("%d %4.4f %s" % rq)
    return out

    
