FROM python:3.8

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch===1.11.0 --no-cache-dir
RUN pip install signatory==1.2.6.1.9.0 --no-cache-dir --force-reinstall
RUN pip install torchvision==0.12.0 --no-cache-dir

ADD configs configs
ADD data data
ADD src src

COPY run.py ./

RUN python run.py --algo RCGAN --dataset AR1
