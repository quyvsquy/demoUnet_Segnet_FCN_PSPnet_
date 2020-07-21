#!/bin/bash
# scp -P 10927 root@0.tcp.ngrok.io:/root/demoUnet_Segnet_FCN_PSPnet_/saveModel/His/unet_0 ./His
PORT=$1
IP=$2
sshpass -p 123456 scp -o StrictHostKeyChecking=no -P $PORT auto_deploy_colab.sh  root@$IP.tcp.ngrok.io:/root
sshpass -p 123456 ssh -o StrictHostKeyChecking=no root@$IP.tcp.ngrok.io -p $PORT
