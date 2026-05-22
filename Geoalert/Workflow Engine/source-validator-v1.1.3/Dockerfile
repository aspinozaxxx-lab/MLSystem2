ARG CI_REGISTRY
FROM ${CI_REGISTRY}/nspd/geoalert/docker/ubuntu-gdal-libspatialindex:1.0.0 AS builder


ARG GITLAB_REGISTRY_TOKEN
ARG GITLAB_ACCESS_TOKEN
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG PIP_SOURCE=https://pypi.org/simple/
ARG CI_SERVER_HOST

RUN pip install --index-url=$PIP_SOURCE \
    aiohttp==3.8.1 \
    boto3==1.21.33 \
    kombu==5.2.4 \
    loguru==0.5.3 \
    pydantic==2.3.0 \
    pyproj==3.2.1 \
    pyyaml==6.0 \
    rasterio==1.2.10 \
    shapely==1.7.1

RUN if [ -z "$GITLAB_ACCESS_TOKEN" ];  \
    then pip install --index-url https://gitlab-ci-token:}/api/v4/projects/723/packages/pypi/simple --no-dependencies we-queue-client[minio]==1.5.2;  \
    else pip install --index-url https://__token__:}/api/v4/projects/723/packages/pypi/simple --no-dependencies we-queue-client[minio]==1.5.2;  \
    fi

COPY data-validator-lib /opt/source-validator/data-validator-lib
WORKDIR /opt/source-validator/data-validator-lib
RUN pip install --no-dependencies .

COPY ./source-validator /opt/source-validator/source-validator

ENV PYTHONPATH "${PYTHONPATH}:/opt/source-validator"
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash
RUN chown -R $USERNAME:$USERNAME /opt/source-validator
WORKDIR /opt/source-validator

# TEST IMAGE
FROM builder AS testing

ARG USERNAME=user
ARG PIP_SOURCE=https://pypi.org/simple/

RUN pip install --index-url=$PIP_SOURCE pytest
COPY ./tests /opt/source-validator/tests
RUN chown -R $USERNAME:$USERNAME /opt/source-validator

USER $USERNAME
CMD ["python3", "-m", "pytest", "./tests/unit"]

# PROD IMAGE
FROM builder AS release

ARG USERNAME=user

# start applictaion
USER $USERNAME
CMD ["python3", "-m",  "source-validator.main"]
