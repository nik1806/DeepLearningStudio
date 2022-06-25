#!/bin/bash

# python3 train.py --data_dir ../complete_dataset/ \
python3 train.py --data_dir ../datasets_opencv/ \
	--preprocess crop \
	--preprocess extreme \
	--data_augs True \
	--num_epochs 1 \
	--batch_size 64 \
	--learning_rate 0.0001 \
	--img_shape "100,50,3"