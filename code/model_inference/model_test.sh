ckpt_path="/path/to/your/checkpoint"
cuda_id="0"
data_path="/path/to/your/data_path"
save_path="/path/to/your/save_path"

python model_test_vicuna_v0.py --ckpt_path $ckpt_path --data_path $data_path ---save_path $save_path -index $cuda_id
