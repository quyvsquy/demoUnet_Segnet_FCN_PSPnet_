apt update \
&& apt install htop vim -y \
&& git clone https://github.com/quyvsquy/demoUnet_Segnet_FCN_PSPnet_ \
&& cd demoUnet_Segnet_FCN_PSPnet_  \
&& git clone https://github.com/circulosmeos/gdown.pl && cd gdown.pl/ && chmod +x gdown.pl && ./gdown.pl https://drive.google.com/file/d/1HP9SM3zT1jKyE0GW9HMeo2eUGyin9lpw/view?usp=sharing ../dataset1.zip \
&& cd .. && unzip dataset1.zip -d dataset1\
&& mkdir -p saveModel/His