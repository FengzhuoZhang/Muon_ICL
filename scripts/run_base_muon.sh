cd src
# export DEBUGPY=1

CUDA_VISIBLE_DEVICES=1 python train_muon.py --config conf/base_muon_tail.yaml


# DEBUGPY=1 python src/train_muon.py --config src/conf/toy_muon.yaml


# DEBUGPY=1 python src/train_muon.py --config src/conf/base_muon_tail.yaml