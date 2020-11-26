# python template

This is a template project for python projects.

## Features

- Pre-commit with `isort`, `yapf`, `flake8`, `trailing-whitespace`, `markdown`, etc.
- Setup script.

## Usage

- Clone the repo and delete the `.git/` folder
- Change the `template` folder to your `projectname`.
- Replace all `template` word in `setup.py` with your `projectname`.

## Development Guide

```bash
# install formatter
pip install yapf

# install pre-commit tool
pip install pre-commit
pre-commit install

# manually check all files
pre-commit run --all-files
```

## Install

```bash
# for development
python setup.py develop

# for deploy
python setup.py install
```
