#!/bin/bash

#block(name=thumos, hours=48, memory=30000, threads=8, gpus=1)

    source /home/richard/anaconda3/bin/activate
    #python train.py
    python inference.py
    cd evalkit
    python eval.py
    cd ..

