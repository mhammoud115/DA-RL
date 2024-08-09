#!/bin/bash


for NPL in 128
do
    for GAMMA in 0.1 0.8 0.9
    do
        for LR in 0.0001
        do
            for BATCHSIZE in 100
            do
                for TOTALTIME in 5000000
                do
                    for VFCOEFF in 0.7 0.8 0.9
                    do
                        for MAXGRAD in 0.8 0.9
                        do
                            sbatch loopSbatch.sh $NPL $GAMMA $LR $BATCHSIZE $TOTALTIME $VFCOEFF $MAXGRAD
                        done
                    done
                done
            done
        done
    done
done
