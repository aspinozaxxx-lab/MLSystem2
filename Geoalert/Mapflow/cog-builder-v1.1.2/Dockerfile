ARG CI_REGISTRY
FROM ${CI_REGISTRY}/nspd/geoalert/docker/ubuntu-gdal-libspatialindex:1.0.0 AS builder

ARG GITLAB_REGISTRY_TOKEN
ARG GITLAB_ACCESS_TOKEN
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG PIP_SOURCE=https://pypi.org/simple/
ARG CI_SERVER_HOST


COPY ./cog_builder /opt/cog_builder/cog_builder
COPY ./requirements.txt /opt/cog_builder/requirements.txt
WORKDIR /opt/cog_builder

RUN pip install --index-url=$PIP_SOURCE -r requirements.txt

RUN if [ -z "$GITLAB_ACCESS_TOKEN" ];  \
    then pip install --index-url https://gitlab-ci-token:}/api/v4/projects/723/packages/pypi/simple --no-dependencies we-queue-client[minio]==1.5.2;  \
    else pip install --index-url https://__token__:}/api/v4/projects/723/packages/pypi/simple --no-dependencies we-queue-client[minio]==1.5.2;  \
    fi


ENV PYTHONPATH "${PYTHONPATH}:/opt/cog_builder"
RUN groupadd --gid $USER_GID $USERNAME && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash
RUN chown -R $USERNAME:$USERNAME /opt/cog_builder
WORKDIR /opt/cog_builder

# TEST IMAGE
FROM builder AS testing

ARG USERNAME=user
ARG PIP_SOURCE=https://pypi.org/simple/

RUN pip install --index-url=$PIP_SOURCE pytest
COPY ./tests /opt/cog_builder/tests
RUN chown -R $USERNAME:$USERNAME /opt/cog_builder

USER $USERNAME
CMD ["python3", "-m", "pytest", "./tests/unit"]

# PROD IMAGE
FROM builder AS release
ARG USERNAME=user

# start applictaion
USER $USERNAME
CMD ["python3", "-m",  "cog_builder.main"]
