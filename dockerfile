ARG IMAGE

FROM ${IMAGE}
LABEL maintainer="Brad Larson"

#USER root

# RUN --mount=type=cache,target=/var/cache/apt \ 
#     apt-get update -y && apt-get install -y --no-install-recommends \
#         openssh-server \
#         autofs \
#         net-tools \
#         iproute2 \
#         pciutils \
#         sudo \
#         tzdata \
#         rsync \
#         tree \
#         htop \
#         git \
#         unzip \
#         expect \
#         apt-utils \
#         software-properties-common \
#         nano \
#         ca-certificates \
#         curl \
#         gnupg \
#         lsb-release \
#         apt-transport-https \
#         daemonize \
#         dbus-user-session \
#         fontconfig && \
#     rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/var/cache/apt \
    pip3 install -r requirements.txt


# RUN mkdir /var/run/sshd
# RUN echo 'root:AD-fgy65r' | chpasswd

# Enable SSH login to root
#RUN sed -i "s/.*PermitRootLogin prohibit-password/PermitRootLogin yes/g" /etc/ssh/sshd_config

EXPOSE 22 3000 5000 6006 8888

# Launch container
#CMD ["/bin/bash"]
CMD ["/usr/sbin/sshd", "-D"]
#CMD ["./run.sh"]