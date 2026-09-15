# WingPrint framework
This repository contains the code corresponding to our WingPrint publication.

Project Website: [https://wingprint.github.io/](https://wingprint.github.io/)

## Datasets

Bats in free flight: [https://hugginface.co/datasets/j-dey/bats-wing-print-free-flight](https://hugginface.co/datasets/j-dey/bats-wing-print-free-flight)

Bat wing scans: [https://huggingface.co/datasets/j-dey/bats-wing-print-scans](https://huggingface.co/datasets/j-dey/bats-wing-print-scans)

## Dependencies

In order to execute the notebooks, create a Python virtual environment:

`python -m venv .venv`

and activate it:

`source .venv/bin/activate`

Finally, you can install all necessary dependencies in your virtual environment:

`pip install -r requirements.txt`

## Free-flight/ML experiments

The free-flight pipeline contains a pre-processing notebook *free_flight_preprocessing.ipynb* that downloads the raw dataset and generates the structured subsets for the experiments.
Afterwards, you can run the experiment notebooks. They correspond to the experiments documented in our WingPrint publication.

## Keypoint matching/SIFT experiments

The keypoint-matching experiment does not have a preprocessing notebook. *keypoint_matching.ipynb* downloads the data and runs the experiment.