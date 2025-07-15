#!/bin/bash

echo "duration,count0,count1" > rabi_data.csv

for t in $(seq 1.0 2.0 400.0); do
    formatted_t=$(printf "%.1f" $t)
    ./GRPCtest 127.0.0.1:2023 60GmonTa 15 1000 1 "TQASM 0.2;\\QREG a[1];\\defcal rabi_test a {frame drive_frame = newframe(a); \\play(drive_frame, cosine_drag($formatted_t, 0.2, 0.0, 0.0)); \\} rabi_test a[0];\\MEASZ a[0];" > tmp.log

    count0=$(grep -A1 'key: "0"' tmp.log | grep 'value:' | awk '{print $NF}')
    count1=$(grep -A1 'key: "1"' tmp.log | grep 'value:' | awk '{print $NF}')

    echo "$t,$count0,$count1" >> rabi_data.csv
    echo "已采集 duration=$t 的数据:  0: $count0 , 1: $count1"
    
    rm tmp.log
done