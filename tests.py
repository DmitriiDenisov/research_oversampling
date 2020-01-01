# pytest tests.py -vv
# pytest tests.py -v
import pytest
import os


def test_1(capsys, caplog):
    os.system('python3 for_tests/multiple_random_point_ND_generalized.py')


def test_2(capsys, caplog):
    os.system('python3 for_tests/multiple_random_point_ND.py')


def test_3(capsys, caplog):
    os.system('python3 for_tests/one_random_point_ND.py')


def test_4(capsys, caplog):
    os.system('python3 main.py')
