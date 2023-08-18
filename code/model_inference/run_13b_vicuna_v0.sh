list=(0 1 2 3 4 5 6 7)
ckpt_path="/path/to/your/checkpoint"
save_path="/path/to/your/save_result"
for i in ${list[@]}; do
  tmux new -d -s "n${i}" "unset http_proxy && unset https_proxy && python inference_13b_v0.py --index ${i} --ckpt_path '$ckpt_path' --save_path '$save_path'; exec bash -i"
done
