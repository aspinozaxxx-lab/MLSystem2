utest:
	docker build --no-cache --target testing --build-arg GITLAB_ACCESS_TOKEN= -t source_validator/test -f Dockerfile .
	docker run --rm source_validator/test python3 -m pytest ./tests/unit
	docker rmi source_validator/test
itest:
	docker build --no-cache --target testing --build-arg GITLAB_ACCESS_TOKEN= -t source_validator/test -f Dockerfile .
	echo "Wait until the service launches"
	docker compose up -d
	sleep 10
	./test.sh
	docker logs source-validator-source-validator-1
	docker compose down
	docker rmi source_validator/test
