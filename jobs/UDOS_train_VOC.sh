conda activate OpenSeg

export lamda=3.0
export delta=10
export data_split_train="voc"
export data_split_test="coco"
export model="maskrcnn_resnet50_fpn"
export data_path="/newfoundland2/tarun/datasets/COCO"

export exp_name=UDOS_${data_split_train}_to_${data_split_test}

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --dataset coco --model ${model} --epochs 9 --lr-steps 6 7 --output-dir ${exp_name} --num-classes 2 --toBinary --lr 0.02 --b 2 --data-split-train ${data_split_train} --data-split-test ${data_split_test} --lamda ${lamda} --delta ${delta} --data-path ${data_path}