task_name=$1    #"ner" or "chunking"
dataset="conll03"

if [ "$task_name" == "chunking" ]; then
   dataset="conll2000"
elif [ "$task_name" != "ner" ]; then
   echo "wrong task name of $task_name, choose between [ner|chunking]"
   exit 1
fi

work_dir=$PWD
data_dir=$work_dir/data/$dataset
output_dir=$work_dir/checkpoints/$task_name

if [ ! -d $work_dir/checkpoints ]; then
    mkdir $work_dir/checkpoints
fi

if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

export PYTHONPATH=$work_dir:$PYTHONPATH
python $work_dir/thumt/bin/trainer.py \
  --model rnnsearch \
  --glove_emb_path $data_dir/eng.glove \
  --bert_emb_path None \
  --input $data_dir/eng.train.src $data_dir/eng.train.trg \
  --output $output_dir \
  --vocabulary $data_dir/vocab.w $data_dir/vocab.t $data_dir/vocab.c \
  --parameters=rnn_cell=DT,save_checkpoint_steps=1000,train_steps=30000,batch_size=1024,max_length=128,constant_batch_size=False,embedding_size=300,learning_rate=8e-4,fine_tuning=False,stack=False,char_embedding_size=128,use_bert=False,use_glove=True,hidden_size=256,dropout=0.5,rnn_dropout=0.3,global_type=mean,global_hidden_size=128,global_transition_num=2,transition_num=2,bert_size=1024
