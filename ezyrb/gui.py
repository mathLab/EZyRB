"""
Utilities for handling the Graphic Unit Interface.

.. todo::
	- Switch to Ttk instead of Tk for a better look of the GUI.
	- Insert a button to decide if plot or not the singular values.
	- Insert a button to decide the path where to save the structures at the end of the procedure.
	- Use grid instead of pack
		
"""

import Tkinter
from tkFileDialog import askopenfilename
from PIL import ImageTk, Image
import ezyrb as ez
import sys
import os
import webbrowser

class Gui(object):
	"""
	The class for the Graphic Unit Interface.

	:cvar string output_name: name of the variable (or output) we want to extract from the solution file.
	:cvar string weights_name: name of the weights to be extracted  from the solution file for the computation
		of the errors. If the solution files does not contain any weight (like volume or area of the cells) 
		the weight is set to 1 for all the cells.
	:cvar string namefile_prefix: path and prefix of the solution files. The files are supposed to be named with
		the same prefix, plus an increasing numeration (from 0) in the same order as the parameter points.
	:cvar string file_format: format of the solution files.
	:cvar string url: url of the github page of PyGeM.
	:cvar tkinter.Label label_new_mu: label where to print the new parameter value.
	:cvar tkinter.Label label_error: label where to print the maximum error on the tesselation.
	:cvar Pod/Interp ezyrb_handler: class for the model reduction. It can be both a Pod and a 
		Interp class (it depends on the is_scalar_switch boolean).
	:cvar bool is_scalar_switch: switch to decide is the output of interest is a scalar or a field.
		
	"""
	
	def __init__(self):
	
		self.root = Tkinter.Tk()
		self.root.title('EZyRB')

		self.output_name  = Tkinter.StringVar()
		self.weights_name = Tkinter.StringVar()
		self.namefile_prefix = Tkinter.StringVar()
		self.file_format = Tkinter.StringVar()
		self.url = 'https://github.com/mathLab/EZyRB'
		self.label_new_mu = None
		self.label_error = None
		self.ezyrb_handler = None
		self.is_scalar_switch = Tkinter.BooleanVar()
		
		self.logo_label = None
		self.img = None
		

	def _start_ezyrb(self):
		"""
		The private method starts the ezyrb algorithm.
		"""
		'''output_name = 'Pressure_drop'
		#output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		#namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.mat'''
		
		if self.is_scalar_switch.get() != True:
			#self.ezyrb_handler = ez.pod.Pod(output_name, weights_name, namefile_prefix, file_format)
			self.ezyrb_handler = ez.pod.Pod(self.output_name.get(), self.weights_name.get(), self.namefile_prefix.get(), self.file_format.get())
		else:
			#self.ezyrb_handler = ez.interpolation.Interp(output_name, namefile_prefix, file_format)
			self.ezyrb_handler = ez.interp.Interp(self.output_name.get(), self.namefile_prefix.get(), self.file_format.get())
		
		self.ezyrb_handler.start()
		self.label_new_mu.configure(text='New parameter value ' + str(self.ezyrb_handler.cvt_handler.mu_values[:,-1]))
		self.label_error.configure(text='Error ' + str(self.ezyrb_handler.cvt_handler.max_error))
		
		
	def _add_snapshot(self):
		"""
		The private method adds a snapshot to the database.
		"""
		self.ezyrb_handler.add_snapshot()
		self.label_new_mu.configure(text='New parameter value' + str(self.ezyrb_handler.cvt_handler.mu_values[:,-1]))
		self.label_error.configure(text='Error ' + str(self.ezyrb_handler.cvt_handler.max_error))
		
		
	def _finish(self):
		"""
		The private method to stop the iterations and save the structures necessary for the online step.
		"""
		self.ezyrb_handler.write_structures()
		self.root.destroy()
		
		
	def _goto_website(self):
		"""
		The private method opens the EZyRB main page on github. 
		It is used for info about EZyRB in the menu.
		"""
		webbrowser.open(self.url)
	
	
	def main(self):
		"""
		The method inizializes and visualizes the window.
		"""
		
		self.logo_panel = Tkinter.Label()
		self.logo_panel.pack(side = "bottom", padx = 5, pady = 5,anchor=Tkinter.SE)
		image = Image.open('readme/logo_EZyRB_small.png')
		image = image.resize((50, 50), Image.ANTIALIAS)
		self.img = ImageTk.PhotoImage(image)
		self.logo_panel.configure(image = self.img)

		text_input_frame = Tkinter.Frame(self.root, relief=Tkinter.GROOVE, borderwidth=1)
		text_input_frame.pack(padx = 5, pady = 5, anchor=Tkinter.W)

		# Buttons 1
		label_prefix = Tkinter.Label(text_input_frame, text="Path and prefix")
		label_prefix.grid(row=0, column=0)
		entry_prefix = Tkinter.Entry(text_input_frame, bd =5, textvariable=self.namefile_prefix)
		entry_prefix.grid(row=0, column=1)

		# Button 2
		label_output = Tkinter.Label(text_input_frame, text="Output of interest")
		label_output.grid(row=1, column=0)
		entry_output = Tkinter.Entry(text_input_frame, bd =5, textvariable=self.output_name)
		entry_output.grid(row=1, column=1)
		
		label_output = Tkinter.Label(text_input_frame, text="Output is a")
		label_output.grid(row=2, column=0)
		switch_frame = Tkinter.Frame(text_input_frame)
		switch_frame.grid(row=2, column=1)
		vtk_radiobutton = Tkinter.Radiobutton(switch_frame, text="Scalar", variable=self.is_scalar_switch, value=True)
		vtk_radiobutton.pack(side=Tkinter.LEFT)
		mat_radiobutton = Tkinter.Radiobutton(switch_frame, text="Field", variable=self.is_scalar_switch, value=False)
		mat_radiobutton.pack(side=Tkinter.RIGHT)
		
		# Button 3
		label_weights = Tkinter.Label(text_input_frame, text="Weight name")
		label_weights.grid(row=3, column=0)
		entry_weights = Tkinter.Entry(text_input_frame, bd =5, textvariable=self.weights_name)
		entry_weights.grid(row=3, column=1)
		
		# Button 4
		format_frame = Tkinter.Frame(text_input_frame)
		format_frame.grid(row=4, column=1)
		
		label_format = Tkinter.Label(text_input_frame, text="Select file format")
		label_format.grid(row=4, column=0)
		
		vtk_radiobutton = Tkinter.Radiobutton(format_frame, text=".vtk", variable=self.file_format, value='.vtk')
		vtk_radiobutton.pack(side=Tkinter.LEFT)
		mat_radiobutton = Tkinter.Radiobutton(format_frame, text=".mat", variable=self.file_format, value='.mat')
		mat_radiobutton.pack(side=Tkinter.RIGHT)
		
		start_frame = Tkinter.Frame(self.root)
		start_frame.pack(padx = 10, pady = 10)
		
		# Start button
		button_run = Tkinter.Button(start_frame, text ="Start EZyRB", command = self._start_ezyrb, bg='#065893', fg='#f19625', font='bold')
		button_run.pack(padx = 5, pady = 5)
		
		display_frame = Tkinter.Frame(self.root, relief=Tkinter.GROOVE, borderwidth=1)
		display_frame.pack()
		
		self.label_new_mu = Tkinter.Label(display_frame, text='Start EZyRB to find the new parameter value')
		self.label_new_mu.pack(padx = 0, pady = 2, anchor=Tkinter.W)
		self.label_error= Tkinter.Label(display_frame, text='Start EZyRB to find the maximum error')
		self.label_error.pack(padx = 0, pady = 0, anchor=Tkinter.W)
		
		chose_frame = Tkinter.Frame(self.root)
		chose_frame.pack(padx = 5, pady = 5)
		
		# Enrich database button
		button_run = Tkinter.Button(chose_frame, text ="Enrich", command = self._add_snapshot, bg='green', fg='white', font='bold')
		button_run.pack(side=Tkinter.LEFT,padx = 5, pady = 5)
		
		# Finish button
		button_finish = Tkinter.Button(chose_frame, text ="Finish", command = self._finish, bg='red', fg='white', font='bold')
		button_finish.pack(side=Tkinter.RIGHT, padx = 5, pady = 5)

		# Menu
		menubar = Tkinter.Menu(self.root)
		helpmenu = Tkinter.Menu(menubar, tearoff=0)
		helpmenu.add_command(label="About...", command=self._goto_website)
		menubar.add_cascade(label="Help", menu=helpmenu)
		self.root.config(menu=menubar)


	def start(self):
	
		self.root.mainloop()

