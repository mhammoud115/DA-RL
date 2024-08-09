#!/bin/bash


for NPL in 128
do
    for GAMMA in 0.05 0.1 0.2 0.25 0.3
    do
        for LR in 0.0001
        do
            for BATCHSIZE in 500 1000
            do
                for TOTALTIME in 5000000
                do
                    for VFCOEFF in 0.7 0.8 0.9
                    do
                        for MAXGRAD in 0.8 0.85 0.9
                        do
                            sbatch loopSbatch.sh $NPL $GAMMA $LR $BATCHSIZE $TOTALTIME $VFCOEFF $MAXGRAD
                        done
                    done
                done
            done
        done
    done
done
