#! /usr/bin/python
# -*- encoding: utf-8 -*-
# prepare all models in a directory for pyLDAvis presentation

import gensim, os, pickle, pyLDAvis, warnings
from IPython.core.display import display, HTML
import pyLDAvis.gensim as gensimvis
warnings.simplefilter("ignore", category=DeprecationWarning)

def prepviz(m, mds='mmds'):
    vis=[]
    mp = "data/%s" % m
    fx = ["%s/%s"%(mp, a) for a in os.listdir(mp) if a.endswith(".model")]
    for f in fx:
        base = f[:-6]
        lda = gensim.models.ldamodel.LdaModel.load(f)
        corpus = pickle.load(open("%s.corpus" % (base)))
        dictionary = gensim.corpora.Dictionary.load("%s.dictionary" % (base))
        vis_data = gensimvis.prepare(lda, corpus, dictionary,mds=mds)
        vis.append(vis_data)
    return vis

def display_with_header(data, header):
    hdata = pyLDAvis.prepared_data_to_html(data)
    hheader = '<h1>%s</h1>' % (header)
    display(HTML(hheader+hdata))
    
