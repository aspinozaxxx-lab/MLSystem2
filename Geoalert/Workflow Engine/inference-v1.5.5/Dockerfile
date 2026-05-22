ARG CI_REGISTRY
FROM ${CI_REGISTRY}/nspd/geoalert/docker/inference-base:1.0.2 as builder


ARG GITLAB_REGISTRY_TOKEN
ARG GITLAB_ACCESS_TOKEN
ARG USERNAME=user
ARG USER_UID=1001
ARG USER_GID=$USER_UID
ARG PIP_SOURCE=https://pypi.org/simple/
ARG CI_SERVER_HOST

# different installation for local and gitlab-ci pipelines
RUN if [ -z "$GITLAB_ACCESS_TOKEN" ];  \
    then pip install --break-system-packages --index-url https://gitlab-ci-token:}/api/v4/projects/723/packages/pypi/simple --no-dependencies we-queue-client[minio]==1.5.2;  \
    else pip install --break-system-packages --index-url https://__token__:}/api/v4/projects/723/packages/pypi/simple --no-dependencies we-queue-client[minio]==1.5.2;  \
    fi

# install Urban submodule
COPY ./inference /opt/inference/inference

COPY ./modules/aeronet_raster /opt/inference/modules/aeronet_raster
COPY ./modules/urban /opt/inference/modules/urban
COPY ./modules/gpdadapter /opt/inference/modules/gpdadapter

WORKDIR /opt/inference/modules/aeronet_raster
RUN pip install --no-dependencies --upgrade --break-system-packages .

WORKDIR /opt/inference/modules/gpdadapter
RUN pip install --no-dependencies --upgrade --break-system-packages .

WORKDIR /opt/inference/modules/urban
RUN pip install --no-dependencies --break-system-packages .

ENV PYTHONPATH "${PYTHONPATH}:/opt/inference"
# Create the user and grant the user permission to the working directory
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash
RUN chown -R $USERNAME:$USERNAME /opt/inference
WORKDIR /opt/inference

# TEST IMAGE
FROM builder AS testing

ARG USERNAME=user
ARG PIP_SOURCE=https://pypi.org/simple/

RUN pip install --index-url=$PIP_SOURCE --break-system-packages pytest
COPY ./tests /opt/inference/tests
RUN chown -R $USERNAME:$USERNAME /opt/inference

USER $USERNAME
CMD ["python3", "-m", "pytest", "./tests/unit"]

# PROD IMAGE
FROM builder AS release
ARG USERNAME=user

# start applictaion
USER $USERNAME
CMD ["python3", "-m",  "inference.main"]
