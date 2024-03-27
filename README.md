# Hopfield Mater
## Requirements
* Python 3.8
* pip (Python package manager)

## Installing Dependencies
In the projects directory, run `pip install -r requirements.txt`. This command must be
executed in order to use the path solver, as it installs all the python dependencies it
uses.

## Running the Application
To run the application, simply `cd` to the project directory and run `python main.py`.
Output will be written to the console, as well as the `test.log` file.
Statistics will be reported to the `report.csv` file

You may have to set the `PYTHONPATH` to point to the project directory if you encounter
import errors

## Generating Average Convergence Data
To generate the raw data to aggregate, run `run_many.sh`

In order to re-generate aggregated stats from raw data, execute `python generate_agg_stats.py`.
Raw data will be searched for in `batch_results` directory. Output will be printed to the screen.

test test
