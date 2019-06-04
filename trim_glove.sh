glove_path=$1
python ./thumt/utils/get_trimmed_glove.py $glove_path ./data/conll03/vocab.w ./data/conll03/eng.glove
python ./thumt/utils/get_trimmed_glove.py $glove_path ./data/conll2000/vocab.w ./data/conll2000/eng.glove
