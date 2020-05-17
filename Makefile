.PHONY: build
build:
	cp .envrc.sample .envrc
	direnv allow
	pipenv install
	pipenv shell

.PHONY: fetch-data
fetch-data:
	kaggle datasets download -d datasnaek/youtube-new
	unzip -d $DATA_DIRECTORY youtube-new.zip
	rm -rf youtube-new.zip

.PHONY: test
test:
	python -m unittest tests/test*.py

.PHONY: lint
lint:
	autopep8 -iv `find . -path "*.py"`
	mypy `find . -path "*.py"`

.PHONY: clean
	echo "Not Implement"