list=(0 1 2 3 4 5 6 7)
for i in ${list[@]}; do
  tmux new -d -s "n${i}" "unset http_proxy && unset https_proxy && python inference_13b_v0.py ${i} ; exec bash -i"
done
