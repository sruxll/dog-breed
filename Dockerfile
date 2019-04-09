FROM ubuntu:18.04

# install the tools and dependencies
RUN apt-get update && apt-get install -y wget curl nano vim make git python3-dev libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pip for Python3
RUN curl https://bootstrap.pypa.io/get-pip.py | python3

# Add the codes into docker image
ADD . /opt/dog_breed/

# install the dependencies
RUN pip3 install -r /opt/dog_breed/requirements.txt

# Set default work directory
WORKDIR /opt/dog_breed/webapp

ENTRYPOINT ["python3"]
CMD ["run_server.py"]
