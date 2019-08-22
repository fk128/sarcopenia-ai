FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get install -y git

ADD ./requirements.txt /
RUN pip install -r /requirements.txt
RUN git clone https://github.com/fk128/midatasets.git && pip install -e midatasets

ADD . /sarcopenia_ai

RUN pip3 install -e /sarcopenia_ai

WORKDIR /sarcopenia_ai
