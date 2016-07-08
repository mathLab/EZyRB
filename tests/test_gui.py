
from unittest import TestCase
import unittest
import ezyrb.gui as gui


class TestGui(TestCase):

	def test_gui_init_string_1(self):
		gui_handler = gui.Gui()
		assert gui_handler.root.title() == 'EZyRB'
		
		
	def test_gui_init_string_2(self):
		gui_handler = gui.Gui()
		assert gui_handler.output_name.get() == ''
		
		
	def test_gui_init_string_3(self):
		gui_handler = gui.Gui()
		assert gui_handler.weights_name.get() == ''
		
		
	def test_gui_init_string_4(self):
		gui_handler = gui.Gui()
		assert gui_handler.namefile_prefix.get() == ''
		
		
	def test_gui_init_string_5(self):
		gui_handler = gui.Gui()
		assert gui_handler.file_format.get() == ''
		
		
	def test_gui_init_string_6(self):
		gui_handler = gui.Gui()
		assert gui_handler.url == 'https://github.com/mathLab/EZyRB'
		
		
	def test_gui_init_none_1(self):
		gui_handler = gui.Gui()
		assert gui_handler.label_new_mu == None
		
		
	def test_gui_init_none_2(self):
		gui_handler = gui.Gui()
		assert gui_handler.label_error == None
		
		
	def test_gui_init_none_3(self):
		gui_handler = gui.Gui()
		assert gui_handler.ezyrb_handler == None
		
		
	def test_gui_init_none_4(self):
		gui_handler = gui.Gui()
		assert gui_handler.logo_label == None
		
		
	def test_gui_init_none_5(self):
		gui_handler = gui.Gui()
		assert gui_handler.img == None
		
		
	def test_gui_init_none_6(self):
		gui_handler = gui.Gui()
		assert gui_handler.is_scalar_switch.get() == False
		
		
	def test_gui_init_all(self):
		gui.Gui()
	
	
	def test_gui_main(self):
		interface = gui.Gui()
		interface.main()
	
		
	'''def test_gui_init_int_1(self):
		gui_handler = gui.Gui()
		assert gui_handler.check_var_1.get() == 0
		
		
	def test_gui_init_int_2(self):
		gui_handler = gui.Gui()
		assert gui_handler.check_var_2.get() == 0
		
		
	def test_gui_init_none_1(self):
		gui_handler = gui.Gui()
		assert gui_handler.label_geo == None
		
		
	def test_gui_init_none_2(self):
		gui_handler = gui.Gui()
		assert gui_handler.label_params == None
		

	def test_gui_init_all(self):
		gui.Gui()
		
		
	def test_gui_main(self):
		interface = gui.Gui()
		interface.main()'''
		
	
