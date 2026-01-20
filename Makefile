.PHONY: build publish clean

build:
	cd packages/rotastellar && python -m build
	cd packages/rotastellar-compute && python -m build
	cd packages/rotastellar-intel && python -m build

publish: build
	cd packages/rotastellar && twine upload dist/*
	cd packages/rotastellar-compute && twine upload dist/*
	cd packages/rotastellar-intel && twine upload dist/*

clean:
	rm -rf packages/*/dist packages/*/build packages/*/*.egg-info
