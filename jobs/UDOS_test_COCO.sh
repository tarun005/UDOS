export delta=10
export data_split_test="all"
export data_path="/newfoundland2/tarun/datasets/UVOdataset"
export load_model="COCO.pth"

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --dataset uvo --model maskrcnn_resnet50_fpn --load_model ${load_model} --num-classes 2 --b 2 --data-split-test ${data_split_test} --toBinary --delta 10 --data-path ${data_path} --detections 300 --test-only --shared
