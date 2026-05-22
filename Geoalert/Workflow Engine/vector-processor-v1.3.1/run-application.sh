#!/usr/bin/env bash

docker-compose up -d database

mvn -f ./backend clean install
docker-compose build --no-cache backend
docker-compose up