glove_path=$1

echo "Triming with the vocab of conll03"
python ./thumt/utils/get_trimmed_glove.py $glove_path ./data/conll03/vocab.w ./data/conll03/eng.glove
echo "Append special symbols: <pad>, <eos>, <unk>, Done!"

echo "Triming with the vocab of conll2000"
python ./thumt/utils/get_trimmed_glove.py $glove_path ./data/conll2000/vocab.w ./data/conll2000/eng.glove
echo "Append special symbols: <pad>, <eos>, <unk>, Done!"
