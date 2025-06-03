"""
This module contains example tests for a Kedro project.
Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py.
"""
from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
import logging


# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality

class TestKedroRun:
    # def test_kedro_run(self):
    #     bootstrap_project(Path.cwd())
    #
    #     with KedroSession.create(project_path=Path.cwd()) as session:
    #         assert session.run() is not None

    def test_kedro_catalog(self):
        bootstrap_project(Path.cwd())

        with KedroSession.create(project_path=Path.cwd(), env="test") as session:
            context = session.load_context()
            catalog = context.catalog
            params = context.params
            assert params["train_hparams"]["epochs"] == 1
            logging.info(context.catalog)

# każdy step na gh ma główne working directory, musi być ustawiony na nazwę tego repo
# pip install -r requirements.txt
# na gh musi być pip install -e .
