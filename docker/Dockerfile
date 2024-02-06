FROM python:3.8

RUN apt-get install -y libcurl4-openssl-dev \
    && apt-get install -y libssl-dev \
    && apt-get update -y \
    && apt-get install -y ffmpeg
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r ./requirements.txt
RUN apt-get install -y locales locales-all

RUN pip3 install lru-dict
RUN pip3 install smbus2

CMD ["/bin/bash"]
