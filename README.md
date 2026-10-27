# NKHDF5

[![PyPI](https://img.shields.io/pypi/v/python-nkhdf5.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/python-nkhdf5.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/python-nkhdf5)][python version]
[![License](https://img.shields.io/pypi/l/python-nkhdf5)][license]

[![Read the documentation at https://python-nkhdf5.readthedocs.io/](https://img.shields.io/readthedocs/python-nkhdf5/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/akhambhati/python-nkhdf5/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/akhambhati/python-nkhdf5/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/python-nkhdf5/
[status]: https://pypi.org/project/python-nkhdf5/
[python version]: https://pypi.org/project/python-nkhdf5
[read the docs]: https://python-nkhdf5.readthedocs.io/
[tests]: https://github.com/akhambhati/python-nkhdf5/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/akhambhati/python-nkhdf5
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

High-performance Python processing pipeline designated to:
* **hdf5writer.py**: Convert raw EDF files (.edf) storing continuous timeseries to BIDS-compatible HDF5 files (.h5)
* **hdf5writer_range_from_bids.py**: Locate, slice and stitch continuous timeseries directly from BIDS-compatible HDF5 files based on time window.

## Features

## Setup and Installation

This project uses **Poetry** for dependency management and virtual environments. Follow the steps below to set it up locally or on the server.

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### 2. Install Poetry
If you do not have Poetry installed, choose one of the methods below:

#### macOS / Linux
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### Windows (PowerShell)
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python3 -
```

#### Alternative (Using pipx - Recommended)
```bash
pip install pipx
pipx install poetry
```

> **Note:** After installation, restart your terminal and run `poetry --version` to ensure it is correctly added to your system path.

### 3. Clone Repository and Environment Setup
Clone the repository, switch to the required branch, configure your environment variables, and install dependencies:

```bash
# Clone the repository and navigate to the project directory
git clone https://github.com/ucsf-npnm/python-nkhdf5.git
cd python-nkhdf5/src/nkhdf5

# Install project dependencies into a virtual environment
poetry install

# Switch to the v5.0 branch
git checkout v5.0

# Initialize environment variables
cp .env.example .env

# Open .env in a code editor to follow in-line comments and save your changes
vim .env
```

### 4. Running the Project (Chang Lab Server)
Avoid running the scripts at the command line! Instead, use the custom shell command named `submit_job` to submit scripts as jobs.

Here is an example of a complete `submit_job` command, including a Poetry command wrapper that automatically targets the path to the Poetry-created virtual environment:

```bash
submit_job -q pia-batch.q -c 8 -m 10 -o /home/user/job_output.txt -n job1 -x $(poetry run which python) /home/user/my_script.py
```
Where:
* `-q pia-batch.q`: Selected job queue. In general, you should only need to use pia.
* `-c 8`: Number of cores for the job. In general, you should not request more than 8 at a time.
* `-m 10`: Total allocated memory (in GB) for the job. In this example, 10 GB of memory is allocated.
* `-o /home/user/job_output.txt`: Full path to printed job output. In this example, stores job output in a text file named job_output.txt inside /home/user directory.
* `-n job1`: (Optional) Job name. In this example, sets the name of the job to job1. Use only letters, numbers and underscores.
* `-x $(poetry run which python)`: Selected executable or environment. In this example, `$(poetry run which python)` is a command wrapper that automatically targets the path to the virtual invironment after running `poetry install` in the previous step. Alternatively, you can specify the complete path to the poetry invironment in your /home directory.
* `/home/user/my_script.py`: The path of the script that you would like to run.

## Usage

Please see the [Command-line Reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [Apache 2.0 license][license],
_NKHDF5_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/akhambhati/python-nkhdf5/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/akhambhati/python-nkhdf5/blob/main/LICENSE
[contributor guide]: https://github.com/akhambhati/python-nkhdf5/blob/main/CONTRIBUTING.md
[command-line reference]: https://python-nkhdf5.readthedocs.io/en/latest/usage.html
