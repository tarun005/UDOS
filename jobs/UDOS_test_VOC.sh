export delta=10
export data_split_test="coco"
export data_path="/datasets/COCO"
export load_model="VOC.pth"

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --dataset coco --model maskrcnn_resnet50_fpn --load_model ${load_model} --num-classes 2 --b 2 --data-split-test ${data_split_test} --toBinary --delta 10 --data-path ${data_path} --detections 300 --test-only
