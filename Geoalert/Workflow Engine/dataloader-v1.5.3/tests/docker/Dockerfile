FROM registry.gitlab.com/geoalert-projects/geoalert-docker/dataloader-base:079ed525

RUN pip install pytest \
                httpretty \
                pytest-asyncio \
                aioresponses

# copy external modules
COPY ./modules /modules

# install external modules
WORKDIR /modules/xyz_loader
RUN pip install --no-dependencies .

WORKDIR /modules/nspd_loader
RUN pip install --no-dependencies .

# copy app
COPY ./app /app
COPY ./tests /tests

# set project workdir
WORKDIR /
ENV PYTHONPATH "${PYTHONPATH}:/app/"