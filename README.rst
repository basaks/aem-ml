===
aem
===


.. image:: https://img.shields.io/pypi/v/aem.svg
        :target: https://pypi.python.org/pypi/aem

.. image:: https://img.shields.io/travis/basaks/aem.svg
        :target: https://travis-ci.com/basaks/aem

.. image:: https://readthedocs.org/projects/aem/badge/?version=latest
        :target: https://aem.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


Introduction
-------------

AEM Assist ML models for Geoscience Australia

AEM assist is a set of machine learning regression algorithm that allows the user to interpret AEM sections from
drillhole observations and/or interpreted segments along the conductivity depth section. AEM assist finds predictive
relationships between the training observations (drillhole and/or interpreted sections) and the conductivity
value which also includes the first vertical derivative of the conductivity.

AEM assists accepts one or more aem interpretation/drillhole information in the form of shapefiles, and their
corresponding conductivity maps. Currently only the following regression algorighms are supported:

    * XGBRegressor
    * GradientBoostingRegressor
    * QuantileGradientBoosting with prediction uncertainties
    * QuantileRandomForestRegressor with prediction uncertainties
    * Quantiles based XGBboostRegression with prediction uncertainties
    * CatBoost

The ML algorithms are set-up in a scikit-learn compatible framework that enables quick and easy adoption of any
scikit learn compatible ML algorithm.


* Free software: Apache Software License 2.0
* Documentation: https://aem.readthedocs.io.


Running
-------

This code uses yaml based configuration files to train and predict. Configuration files are available in the *configs*
directory.

Very briefly, the workflow consists of the following steps:

1. learn an ML model: `aem learn -c configs/xgboost.yaml`
2. predict using this model: `aem predict -c configs/xgboost.yaml --model-type learn`
3. optionally run oos validation: `aem validate -c configs/xgboost.yaml`
4. optimise the model parameters: `aem optimise -c configs/xgboost.yaml `


Installation
------------

The following system dependencies are required by aem:

- `Python <https://www.python.org/downloads/>`_, versions 3.6, 3.7 or 3.8.

Installing Python3:

1. Start by updating package list on terminal and installing preqs:
$ sudo apt update
$ sudo apt install software-properties-common

2. Add the deadsnakes PPA to sources list:
$ sudo add-apt-repository ppa:deadsnakes/ppa

3. Install Python 3.7:
$ sudo apt install python3.7

4. Verify Installation:
$ python3.7 --version


Python dependencies for aem-ml are::
   pip==19.2.3
   bump2version==0.5.11
   wheel~=0.35
   watchdog==0.9.0
   flake8==3.7.8
   tox==3.14.0
   coverage==5.5
   Sphinx==1.8.5
   twine==1.14.0
   pytest==6.2.4
   ipython==7.24.1
   scikit-optimize==0.8.1
   ghp-import==2.0.1
   sphinxcontrib-programoutput==0.17
   pytest-cov==2.12.1
   codecov==2.1.11

Python Environment
-------------------

Install `virtualenv` and `virtualenvwrapper` `for your python version. <https://gist.github.com/basaks/b33ea9106c7d1d72ac3a79fdcea430eb>`_

Installing virtualenv:
1. Creating special directory for virtualenvs:
$ mkdir .virtualenv

2. Installing pip for Python3
$ sudo apt install python3-pip
$ pip3 --version

3. Install virtualenv via pip3
$ pip3 install virtualenv
$ which virtualenv

4. Install virtualenvwrapper via pip3:
$ pip3 install virtualenvwrapper

5. Modify .bashrc using vim editor
$ sudo apt install vim
$ vim .bashrc

6. Navigate to the bottom of the .bashrcfile, press "i" and add:
#Virtualenvwrapper settings:
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_VIRTUALENV=/home/your_username/.local/bin/virtualenv
source ~/.local/bin/virtualenvwrapper.sh

Once you are done press esq key and type :wq and press enter. Restart terminal.

7. Create virtualenv
$ mkvirtualenv "NewEnv"

8. List virtualenvironments with :
$ workon

9. Activate specific environment using:

$ workon "Name of env"

Then complete the following steps:

.. code-block:: python

Navigate to aem-ml
   mkdir -p python3.7 aemml
   pip install -r requirements.txt -r requirements_dev.txt

Once everything in installed, run the tests:

.. code-block:: python

    python setup.py develop
    pytest tests/


Credits
-------
John Wilford
Sudipta Basak

