#!/bin/bash
docker build -t dataloader:test -f tests/docker/Dockerfile .
# build and run mock server
# it handles http:127.0.0.1/tile/z/x/y request, sleeps for z seconds and answers 401 if x==401 and 403 if x==403;
# answers 200 with image otherwise
# mock server output must contain exactly one 401 response (correct behavior) and multiple 403 responses
docker build -t mock_tile_server -f tests/mock_server/Dockerfile ./tests/mock_server
command=$1
if [[ $command == "it" ]]; then
  # only integration tests
  docker compose -f ./tests/it/docker-compose.yaml up --abort-on-container-exit; cd tests/it; docker compose down
elif [[ $command == "unit" ]]; then
  # both unit tests and integration tests
  docker run --env-file ./tests/docker/.env --rm -it dataloader:test pytest tests/unit
elif [[ $command == "manual" ]]; then
  # manual tests require looking into server's logs to see that only one request is receiced
  # the compose command must output logs for both mock server and dataloader
  docker compose -f ./tests/manual/docker-compose.yaml up --abort-on-container-exit; cd tests/manual; docker compose down
elif [[ $command == "memory" ]]; then
  docker run --env-file ./tests/docker/.env --rm -it -v /home/trekin/projects/workflow-engine-project/dataloader/test_data:/output dataloader:test pytest -v tests/manual/xyz_loader/memory_profiling.py
else
  echo COMMAND NOT SPECIFIED! Use \"it\" \"unit\" or \"manual\"
fi
