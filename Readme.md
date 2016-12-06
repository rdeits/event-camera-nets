# Installation

First, create a new virtualenv (if you haven't already):

	virtualenv venv --python=python3

### Activate the environment:

	source venv/bin/activate


### Install prereqs:

On Ubuntu, you will at least need to do:

    sudo apt install libdf5-dev

### Install required packages:

	pip3 install -r requirements.txt

Install tensorflow from https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html

### Activate the eventcnn package for development:

	python3 setup.py develop



