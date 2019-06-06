task_name=$1    # "ner" or "chunking"
dataset="conll03"
test_type=$2    # "testa" or "testb" for "ner";  "testb" for chunking

if [ "$task_name" == "chunking" ]; then
   dataset="conll2000"
elif [ "$task_name" != "ner" ]; then
   echo "wrong task name of $task_name, choose between [ner|chunking]"
   exit 1
fi

if [ "$test_type" != "testa" ] && [ "$test_type" != "testb" ]; then
    echo "wrong test name of $test_type, choose between [testa|testb]"
    exit 2
fi

work_dir=$PWD
code_dir=$work_dir/thumt
data_dir=$work_dir/data/$dataset
result_dir=$work_dir/results/$task_name

if [ ! -d $work_dir/results ]; then
    mkdir $work_dir/results
fi

if [ ! -d $result_dir ]; then
    mkdir $result_dir
fi

export PYTHONPATH=$work_dir:$PYTHONPATH

init_step=1000
step_size=1000
total_step=30000
for idx in `seq $init_step $step_size $total_step` 
do
    echo model_checkpoint_path: \"model.ckpt-$idx\" > $work_dir/checkpoints/$task_name/checkpoint
    echo decoding with $task_name/checkpoint-$idx
    python $code_dir/bin/translator.py \
        --models rnnsearch \
        --checkpoints $work_dir/checkpoints/$task_name \
        --input $data_dir/eng.$test_type.src \
        --glove_emb_path $data_dir/eng.glove \
        --bert_emb_path None \
        --output $result_dir/$test_type.out.$idx \
        --vocabulary $data_dir/vocab.w $data_dir/vocab.t $data_dir/vocab.c \
        --parameters=decode_batch_size=64

    python $data_dir/get_score.py \
           -s $data_dir/eng.$test_type.src \
           -g $data_dir/eng.$test_type.trg \
           -p $result_dir/$test_type.out.$idx \
           -r $result_dir/$test_type.mod.$idx \
          > $result_dir/$test_type.eval.$idx

done



