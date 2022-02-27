# This Dockerfile sets up part of the development environment for COS-POMDP;
# Please read the instructions at for steps to create an image from this
# Dockerfile and subsequent steps to setup the environment inside the container.
FROM ubuntu:20.04

# Install software
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y emacs
RUN apt-get install -y sudo
RUN apt-get install -y python-pip
RUN apt-get install -y net-tools
RUN apt-get install -y iproute2
RUN apt-get install -y iputils-ping
RUN apt-get install -y openssh-client openssh-server
RUN apt-get install -y gdb
RUN apt-get install -y mlocate

# create a user
ARG hostuser=kaiyu

RUN adduser --disabled-password --gecos '' $hostuser
RUN adduser $hostuser sudo
# Ensure sudo group users are not asked for a p3assword when using sudo command
# by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> \
/etc/sudoers

USER $hostuser
WORKDIR /home/$hostuser
ENV HOME=/home/$hostuser
RUN mkdir $HOME/repo

# Different shell color
RUN echo "export PS1='\[\033[01;31m\]\u@\h\[\033[00m\]:\[\033[01;33m\]\w\[\033[00m\]$ '" >> $HOME/.bashrc

# print some info on start
RUN echo "echo -e 'Welcome! You are now in a docker container ().'" >> $HOME/.bashrc
RUN echo "echo -e \"Docker ID: $(basename $(cat /proc/1/cpuset))\"" >> $HOME/.bashrc
CMD ["bash"]
