cd src
# export DEBUGPY=1

# CUDA_VISIBLE_DEVICES=1 python train_muon.py --config conf/base_muon_tail.yaml --training.adam_lr 5.0 --out_dir /home/aiops/zhangfz/Muon_linear_regression/Muon_ICL/logs/linear_regression_tail/adam_lr_search



SGD_LEARNING_RATES=(5e-3 2e-3 1e-3 5e-4 2e-4 1e-4 5e-5 2e-5)
SEEDS=(42 43 44 45 46) #43 44 45 46

for SEED in "${SEEDS[@]}"; do
    for SGD_LR in "${SGD_LEARNING_RATES[@]}"; do
        echo ""
        echo "======================================================================"
        echo "  STARTING RUN: Mode=$MODE, SGD_LR=$SGD_LR, Seed=$SEED"
        echo "======================================================================"

        CUDA_VISIBLE_DEVICES=1 python train_muon.py \
            --config conf/base_muon_tail.yaml \
            --training.optimizer "muon" \
            --training.adam_lr 1e-4 \
            --training.muon_lr "$SGD_LR" \
            --training.seed "$SEED" \
            --out_dir /home/aiops/zhangfz/Muon_linear_regression/Muon_ICL/logs/linear_regression_tail/muon_lr_search
        


        
        echo ""
        echo "----------------------------------------------------------------------"
        echo "  FINISHED RUN: Mode=$MODE, SGD_LR=$SGD_LR, Seed=$SEED"
        echo "----------------------------------------------------------------------"
        echo ""
        
        
    done
done


# DEBUGPY=1 python src/train_muon.py --config src/conf/toy_muon.yaml


# DEBUGPY=1 python src/train_muon.py --config src/conf/base_muon_tail.yaml