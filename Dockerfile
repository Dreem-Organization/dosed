FROM floydhub/pytorch:1.0.0-gpu.cuda9cudnn7-py3.38

RUN pip install pip --upgrade
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

WORKDIR /workspace