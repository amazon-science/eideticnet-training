#!/bin/bash

optimizer="SGD"
lr=0.3
momentum=0.9
reduce_lr=4
pruning_type=taylor
stop_pruning_threshold=0.01
max_recovery_epochs=100
batch_size=256

#for epochs in 1
#                --hold-out \
#                --hold-out-num-tasks 3 \
for early_stopping_patience in 10
do
    for pruning_step_size in 0.025
    do
        for weight_decay in 0.001
        do
            time python ensemble.py \
                --num-training-subsets 100 \
                --early-stopping-patience $early_stopping_patience \
                --pruning-step-size $pruning_step_size \
                --weight-decay $weight_decay \
                --reduce-learning-rate-before-pruning $reduce_lr \
                --optimizer $optimizer \
                --lr $lr \
                --momentum $momentum \
                --pruning-type $pruning_type \
                --stop-pruning-threshold $stop_pruning_threshold \
                --max-recovery-epochs $max_recovery_epochs \
                --forward-transfer \
                --batch-size $batch_size \
                --in-channels 3
        done
    done
done
