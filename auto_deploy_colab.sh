# scp -P 18376 auto_deploy_colab.sh  root@2.tcp.ngrok.io:/root

apt install vim -y
git clone https://github.com/quyvsquy/demoUnet_Segnet_FCN_PSPnet_
cd demoUnet_Segnet_FCN_PSPnet_
wget https://media.githubusercontent.com/media/quyvsquy/demoUnet_Segnet_FCN_PSPnet_/master/dataset1.zip
unzip dataset1.zip.1
mkdir saveModel
vi test.py