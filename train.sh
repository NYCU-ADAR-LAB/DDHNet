model='DLRM'
fc_sparsity='0'
embedding_size=64
bot_mlp='13-512-256-64'
top_mlp='1024-512-256'

interaction_op='mix'
compressed_dim=16
expansion_factor='1-1'
mix_act='none'

lr_decay='cosine'
warm=20
optimizer_type='adam'
weight_decay=6e-7

#without any dataset preprocess
train_csv_file="/work/twsugkm569/data/kaggle/dataset_mirror_normalize9925/train.csv"
valid_csv_file="/work/twsugkm569/data/kaggle/dataset_mirror_normalize9925/valid.csv"

time=$(date '+%Y%m%d%H%M%S')
task="${model}_IP8_${embedding_size}_FC${fc_sparsity}_${interaction_op}to${compressed_dim}_act_${mix_act}_optim_${optimizer_type}_l2_${weight_decay}_lr_decay_${lr_decay}_warm${warm}_mirror_normalize9925_DIDL_2MB_pdw"
echo $task
save_path="./runs/${task}"
log_file="./logs/${task}_${time}.txt"

dlrm_pt_bin='python main.py'
#
$dlrm_pt_bin \
--model=$model \
--fc_sparsity=$fc_sparsity \
--interaction_op=$interaction_op \
--compressed_dim=$compressed_dim --expansion_factor=$expansion_factor --mix_act=$mix_act \
--embedding_size=$embedding_size \
--bottom_mlp=$bot_mlp --top_mlp=$top_mlp \
--n_epochs=25 \
--optimizer_type=$optimizer_type --weight_decay=$weight_decay \
--lr_decay=$lr_decay --warm=$warm \
--save_path=$save_path \
--test_batch_size=131072 \
--train-csv-file=$train_csv_file \
--valid-csv-file=$valid_csv_file \
2>&1 | tee $log_file
