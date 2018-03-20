from unittest import TestCase
import unittest
import numpy as np
import filecmp
import os

from ezyrb.parametricspace import ParametricSpace


class TestParametricSpace(TestCase):
    def test_init(self):
        with self.assertRaises(NotImplementedError):
            ParametricSpace()

    def test_call(self):
        with self.assertRaises(NotImplementedError):
            ParametricSpace()(0.0)
