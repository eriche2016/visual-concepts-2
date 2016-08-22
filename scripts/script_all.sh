# if dir not exists, then create directory, which will be used for writing splits and labels for training to be used for Caffe


rm *.pyc

if [ ! -d output/v1 ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir output/v1
fi

# modify the conditional statements to run corresponding commands
if [ 1 -eq 0 ]; then 
python run_mil.py --task compute_targets \
  --train_dir output/v1/ --write_labels 1 --write_splits 1 \
  --train_set train --val_set valid1 \
  --vocab_file vocabs/vocab_train.pkl 
fi

# train our model 
if [ 1 -eq 0 ]; then 
# Command to launch the training
GLOG_logtostderr=1 caffe/build/tools/caffe.bin train -gpu 1 \
  -model output/v1/mil_finetune.prototxt \
  -solver output/v1/mil_finetune_solver.prototxt \
  -weights ../caffe-data/vgg_16_full_conv.caffemodel 2>&1 \
  | tee output/v1/training.log # stdout:file descriptor:1, stderr: file descriptor:2
                               # here: 2>&1: redirects stderr to stdout, |tee log.txt: combines 
                               # both stdout(standard output) and stderr(standard error) and save 
                               # to the same file named log.txt 
fi

# testing
if [ 1 -eq 1 ]; then 
# Testing the pre-trained model on valid2 
python run_mil.py --task test_model \
  --prototxt_deploy output/vgg/mil_finetune.prototxt.deploy \
  --model output/vgg/snapshot_iter_240000.caffemodel \
  --vocab_file vocabs/vocab_train.pkl \
  --test_set valid2 \
  --gpu 1
fi

if [ 1 -eq 1 ]; then 
# Testing the pre-trained model on valid1 
python run_mil.py --task test_model \
  --prototxt_deploy output/vgg/mil_finetune.prototxt.deploy \
  --model output/vgg/snapshot_iter_240000.caffemodel \
  --vocab_file vocabs/vocab_train.pkl \
  --test_set valid1 --gpu 1
fi 

if [ 1 -eq 1 ]; then 
# Testing the pre-trained model on test
python run_mil.py --task test_model \
  --prototxt_deploy output/vgg/mil_finetune.prototxt.deploy \
  --model output/vgg/snapshot_iter_240000.caffemodel \
  --vocab_file vocabs/vocab_train.pkl \
  --test_set test --gpu 1
fi 

# evaluating the model on valid1 dataset 
if [ 1 -eq 1 ]; then 
# Benchmarking the pre-trained model
python run_mil.py --task eval_model --gpu 1 \
  --prototxt_deploy output/vgg/mil_finetune.prototxt.deploy \
  --model output/vgg/snapshot_iter_240000.caffemodel \
  --vocab_file vocabs/vocab_train.pkl \
  --test_set valid1
fi 


# generating txt file 
# generating prediction txt on valid1, valid2, and test
if [ 1 -eq 1 ]; then 
# Generating output txt files for pre-trained model
python run_mil.py --task output_words --gpu 1 \
  --model output/vgg/snapshot_iter_240000.caffemodel \
  --test_set valid1 \
  --calibration_set valid1 \
  --vocab_file vocabs/vocab_train.pkl 
fi 

if [ 1 -eq 1 ]; then 
# Generating output txt files for pre-trained model
python run_mil.py --task output_words --gpu 1 \
  --model output/vgg/snapshot_iter_240000.caffemodel \
  --test_set valid2 \
  --calibration_set valid1 \
  --vocab_file vocabs/vocab_train.pkl 
fi 

if [ 1 -eq 1 ]; then 
# Generating output txt files for pre-trained model
python run_mil.py --task output_words --gpu 1 \
  --model output/vgg/snapshot_iter_240000.caffemodel \
  --test_set test \
  --calibration_set valid1 \
  --vocab_file vocabs/vocab_train.pkl 
fi 
