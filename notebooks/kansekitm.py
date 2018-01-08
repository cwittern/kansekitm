#! /usr/bin/python
# -*- encoding: utf-8 -*-
# This can be imported into the notebook to hide this noise.
import gensim, mmseg, os, codecs
from collections import defaultdict
frequency=defaultdict(int)
kstm_base='/home/chris/00scratch/kansekitm'
corpus_base='%s/corpus/zztj' % (kstm_base)
t = mmseg.Dictionary()
t.load_chars('%s/dic/chars.dic' % (kstm_base ))
t.load_words('%s/dic/words.dic' % (kstm_base ))
files = os.listdir(corpus_base)
files.sort()
of=codecs.open("%s/out.txt" % (corpus_base)  , "w", "utf-8")
for f in files:
    if not f.startswith("zztj"):
        continue
    of.write("# file: %s\n" % (f))
    print "%s/%s" % (corpus_base, f)
    for line in codecs.open("%s/%s" % (corpus_base, f), 'r', 'utf-8'):
        if line[0] in ['*', '#']:
            continue
        l_out=[]
        for l in line.split():
            if "@" in l:
                l_out.append(l.split('@')[-1])
            else:
                algor = mmseg.Algorithm(l)
                l_out.extend([tok.text for tok in algor])
        of.write("%s\n" % (" ".join(l_out)))
        for token in l_out:
            frequency[token] += 1
of.close()
