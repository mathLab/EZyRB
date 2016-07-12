
from unittest import TestCase
import unittest
import ezyrb.online as on
import numpy as np
import filecmp
import os


class TestOnline(TestCase):


	def test_online_attributes_01(self):
		mu_value = np.array([.0, .0])
		output_name = 'Pressure'
		online_handler = on.Online(mu_value, output_name)
		np.testing.assert_array_almost_equal(online_handler.mu_value, mu_value)
		
		
	def test_online_attributes_02(self):
		mu_value = np.array([.0, .0])
		output_name = 'Pressure'
		online_handler = on.Online(mu_value, output_name)
		assert online_handler.output_name == 'Pressure'
		
		
	def test_online_attributes_default_03(self):
		mu_value = np.array([.0, .0])
		output_name = 'Pressure'
		online_handler = on.Online(mu_value, output_name)
		assert online_handler.directory == './'
		
		
	def test_online_attributes_03(self):
		mu_value = np.array([.0, .0])
		output_name = 'Pressure'
		directory = 'tests/'
		online_handler = on.Online(mu_value, output_name, directory=directory)
		assert online_handler.directory == 'tests/'
		
		
	def test_online_attributes_default_04(self):
		mu_value = np.array([.0, .0])
		output_name = 'Pressure'
		online_handler = on.Online(mu_value, output_name)
		assert online_handler.is_scalar == True
		
		
	def test_online_attributes_04(self):
		mu_value = np.array([.0, .0])
		output_name = 'Pressure'
		is_scalar = False
		online_handler = on.Online(mu_value, output_name, is_scalar=is_scalar)
		assert online_handler.is_scalar == False
		
		
	def test_online_run_01(self):
		mu_value = np.array([.0, .0])
		output_name = 'Pressure'
		directory = 'tests/test_datasets/'
		online_handler = on.Online(mu_value, output_name, directory=directory)
		online_handler.run()
		np.testing.assert_almost_equal(online_handler.output, 13.21895647)
		
		
	def test_online_run_02(self):
		mu_value = np.array([.0, .0])
		output_name = 'Pressure'
		directory = 'tests/test_datasets/'
		is_scalar = False
		online_handler = on.Online(mu_value, output_name, directory=directory, is_scalar=is_scalar)
		online_handler.run()
		
		
		
		
		
		
