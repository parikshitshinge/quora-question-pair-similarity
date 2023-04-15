Question pair similarity
==============================

Predicts whether two questions or sentences are similar to each other. Helps in deduplication of questions asked in forums like Quora, Reddit etc. Uses NLP techniques to build features.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    |
    ├── data
    │   ├── source         <- The source data sets in case the source files are downloaded from source systems.
    │   ├── raw            <- The original, immutable data dump.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── external       <- Data from third party sources.
    |
    ├── logs               <- Logs generated when program is running
    │
    ├── models             <- Trained and serialized models, model predictions, model summaries, or vectorizers
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    |
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── app.py         <- Initiator for flask app
    │   ├── exception.py   <- Custom exception handler
    │   ├── logger.py      <- Custom logger
    │   ├── utils.py       <- Set of custom utilities functions
    │   │
    │   ├── data           <- Scripts to extract data and transform data
    │   │   ├── data_ingestion.py
    │   │   └── data_transformation.py
    │   │
    │   ├── pipelines      <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── train_pipeline.py
    │   │   └── predict_pipeline.py
    │   │
    │   └── templates      <- Web pages to host to take real time inputs and provide predictions
            ├── index.html
            └── home.html


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
