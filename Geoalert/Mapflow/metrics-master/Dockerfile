ARG CI_REGISTRY
FROM ${CI_REGISTRY}/nspd/geoalert/docker/ubuntu-gdal-libspatialindex:1.0.0 AS builder

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG PIP_SOURCE=https://pypi.org/simple/

RUN apt-get update && apt-get install -y \
    wget \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libxext6

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install \
    opencv-python==4.2.0.34 \
    shapely==1.8.5.post1 \
    rasterio==1.2.10 \
    rtree==0.9.7 \
    aeronet[all]==0.2.2 \
    geopandas==0.13.2 \
    jupyterlab==2.2.8 \
    ipywidgets==7.5.1 \
    cython==3.0.11 \
    matplotlib==3.7.5 \
    && jupyter nbextension enable --py widgetsnbextension

# Jupyter
## Download and install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh
ENV PATH /opt/conda/bin:$PATH

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD: > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Add configuration file for Jupyter and Entrypoint script
COPY --chown=user:user jupyter_notebook_config.py /home/user/.jupyter/
# Add notebooks
COPY --chown=user:user ./src /home/user/src

#COPY run.sh /opt/run/

# Make entrypoint script executable
# Add conda path for user
#RUN chmod +x /opt/run/run.sh && \
#    echo "export PATH='/opt/conda/bin:${PATH}'" >> /home/user/.bashrc

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

WORKDIR /home/user
CMD jupyter lab
