.PHONY: build
build:
  cp .envrc.sample .envrc
  pipenv install

.PHONY: load-data
load-data:
	kaggle datasets download -d datasnaek/youtube-new
	unzip -d youtube-new youtube-new.zip
	rm -rf youtube-new.zip

.PHONY
test:
	pipenv run lint
	pipenv run mypy
	pipenv run test
