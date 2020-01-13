#!/bin/bash
for i in {1..100} ; do
    mkdir batch_results/${i}
    python main.py
    mv report.csv test.log batch_results/${i}
done
