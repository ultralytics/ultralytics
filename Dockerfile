FROM ubuntu:22.04

WORKDIR /
RUN apt update && apt install -y git
RUN git clone https://github.com/axelera-ai-hub/voyager-sdk.git

RUN DEBIAN_FRONTEND=noninteractive apt install -y python3-full python3-venv

RUN apt update && apt install sudo

RUN python3 -m venv env
