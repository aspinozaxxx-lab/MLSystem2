ci-login:
	@docker login -u ${CI_REGISTRY_USER} -p ${CI_REGISTRY_PASSWORD} ${CI_REGISTRY}

clean:
	@docker rmi $(shell docker images -qf dangling=true) || true

reset-db:
	@docker-compose kill database || true
	@docker-compose rm -vf database || true
	@docker volume rm wm_database || true
	@docker-compose up -d database

up:
	@docker-compose up -d

up-build:
	@docker-compose up -d --build

stop-rm: kill
	@docker-compose rm -f

restart: stop-rm up

ps:
	@docker-compose ps

kill:
	@docker-compose kill

down:
	@docker-compose down -v

logs:
	@docker-compose logs -f --tail=300 backend

setup-docker-env:
	export CI_COMMIT_SHORT_SHA=latest
	export CI_REGISTRY_IMAGE=mapflow-api