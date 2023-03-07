# use one of the images from this repository: https://github.com/centreborelli/ipol-docker-images/
FROM registry.ipol.im/ipol:v1-py3.8

# install additional debian packages
COPY .ipol/packages.txt packages.txt
RUN apt-get update && apt-get install -y $(cat packages.txt) && rm -rf /var/lib/apt/lists/* && rm packages.txt

# copy the requirements.txt and install python packages
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt && rm requirements.txt


ENV bin /workdir/bin/
RUN mkdir -p $bin/tmp


WORKDIR $bin

RUN echo "Hi I'm here"

# download the weights

# RUN mkdir -p weights && pip3 install gdown==4.5.1

# RUN gdown 12sLyqBp0VFwdpA-oZLdIOkOTkz_ZnIhV -O weights/FFraw.tar \
#     && gdown 1X0-NYT8KPursLZZdxduRQju6E52hauV0 -O weights/FFc23.tar \
#     && wget https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth \
#     && mv adv-efficientnet-b4-44fb3a87.pth weights/ \
#     && wget https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip \
#     && unzip retinaface_resnet50_2020-07-20-f168fae3c.zip -d weights/ && rm retinaface_resnet50_2020-07-20-f168fae3c.zip

# copy the code to $bin
COPY . .



WORKDIR $bin/jm
RUN make -j ldecod



# the execution will happen in the folder /workdir/exec
# it will be created by IPOL

# some QoL tweaks
ENV PYTHONDONTWRITEBYTECODE 1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python
ENV PATH $bin:$PATH

# $HOME is writable by the user `ipol`, but 
ENV HOME /home/ipol
# chmod 777 so that any user can use the HOME, in case the docker is run with -u 1001:1001
RUN groupadd -g 1000 ipol && useradd -m -u 1000 -g 1000 ipol -d $HOME && chmod -R 777 $HOME
USER ipol