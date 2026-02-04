FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

ENV MY_APT='apt -o "Acquire::https::Verify-Peer=false" -o "Acquire::AllowInsecureRepositories=true" -o "Acquire::AllowDowngradeToInsecureRepositories=true" -o "Acquire::https::Verify-Host=false"'

RUN $MY_APT update && $MY_APT install -y curl gcc g++ nano sudo libgeos-dev libeccodes-dev libeccodes-tools git vim libtiff5 openssh-server


COPY requirements.txt /app/requirements.txt
RUN set -eux \
    && pip install --default-timeout=100 -r /app/requirements.txt

ARG USERNAME
ARG GROUPNAME
ARG USER_UID
ARG USER_GUID
ARG HOME_DIR
ARG NODE_EXTRA_CA_CERTS

RUN set -eux && groupadd --gid $USER_GUID $GROUPNAME \
    # https://stackoverflow.com/questions/73208471/docker-build-issue-stuck-at-exporting-layers
    && mkdir -p $HOME_DIR && useradd -l --uid $USER_UID --gid $USER_GUID -s /bin/bash --home-dir $HOME_DIR --create-home $USERNAME \
    && chown $USERNAME:$GROUPNAME $HOME_DIR \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && echo "$USERNAME:$USERNAME" | chpasswd

WORKDIR $HOME_DIR
