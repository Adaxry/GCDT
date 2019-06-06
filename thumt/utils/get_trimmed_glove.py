import sys 
import codecs

with codecs.open(sys.argv[1], "r", encoding="utf-8") as emb_f, \
     codecs.open(sys.argv[2], "r", encoding="utf-8") as vocab_f, \
     codecs.open(sys.argv[3], "w", encoding="utf-8") as glove_f:

    vocab = set([word.strip() for word in vocab_f])
    done_set = set()
    for line in emb_f:
        try:
            vector = line.strip().split()
            if len(vector) != 301:
                continue
            else:
                word = line.split()[0]
        except:
            continue
        if word in vocab and not word in done_set:
            done_set.add(word)
            glove_f.write(line)
