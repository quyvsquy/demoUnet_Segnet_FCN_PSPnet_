#!/bin/bash
PORT=$1
IP=$2
sshpass -p 123456 ssh -o StrictHostKeyChecking=no root@$IP.tcp.ngrok.io -p $PORT