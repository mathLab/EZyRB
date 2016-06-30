"""
Utilities for handling the Graphic Unit Interface.

.. todo::
	Switch to Ttk instead of Tk for a better look of the GUI
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
	:cvar Tkinter.Label label_new_mu: label where to print the new parameter value.
	:cvar Tkinter.Label label_error: label where to print the maximum error on the tesselation.
	:cvar Pod pod_handler: class for the proper orthogonal decomposition.
	
	::todo:
		Insert a button to decide if plot or not the singular values.
		Insert a button to decide the path where to save the structures at the end of the procedure.
	
	
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
		self.pod_handler = None
		
		self.logo_label = None
		self.img = None
		

	def _start_ezyrb(self):
		"""
		The private method starts the ezyrb algorithm.
		"""
		self.pod_handler = ez.pod.Pod(self.output_name, self.weights_name, self.namefile_prefix, self.file_format)
		'''output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk
		self.pod_handler = ez.pod.Pod(output_name, weights_name, namefile_prefix, file_format)'''
		self.pod_handler.start()
		self.label_new_mu.configure(text='New parameter value ' + str(self.pod_handler.cvt_handler.mu_values[:,-1]))
		self.label_error.configure(text='Error ' + str(self.pod_handler.cvt_handler.max_error))
		
		
	def _add_snapshot(self):
		"""
		The private method adds a snapshot to the database.
		"""
		self.pod_handler.add_snapshot()
		self.label_new_mu.configure(text='New parameter value' + str(self.pod_handler.cvt_handler.mu_values[:,-1]))
		self.label_error.configure(text='Error ' + str(self.pod_handler.cvt_handler.max_error))
		
		
	def _finish(self):
		"""
		The private method to stop the iterations and save the structures necessary for the online step.
		"""
		self.pod_handler.write_structures()
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

		text_input_frame = Tkinter.Frame(self.root)
		text_input_frame.pack(anchor=Tkinter.W)

		# Buttons 1
		label_prefix = Tkinter.Label(text_input_frame, text="Path and prefix")
		label_prefix.pack(anchor=Tkinter.NW)
		entry_prefix = Tkinter.Entry(text_input_frame, bd =5, textvariable=self.namefile_prefix)
		entry_prefix.pack()

		# Button 2
		label_output = Tkinter.Label(text_input_frame, text="Output of interest")
		label_output.pack(anchor=Tkinter.W)
		entry_output = Tkinter.Entry(text_input_frame, bd =5, textvariable=self.output_name)
		entry_output.pack()
		
		# Button 3
		label_weights = Tkinter.Label(text_input_frame, text="Weight name")
		label_weights.pack(anchor=Tkinter.SW)
		entry_weights = Tkinter.Entry(text_input_frame, bd =5, textvariable=self.weights_name)
		entry_weights.pack()
		
		# Button 4
		format_frame = Tkinter.Frame(self.root)
		format_frame.pack(anchor=Tkinter.W)
		
		label_format = Tkinter.Label(text_input_frame, text="Select file format")
		label_format.pack(anchor=Tkinter.W)
		
		vtk_radiobutton = Tkinter.Radiobutton(format_frame, text=".vtk", variable=self.file_format, value='.vtk')
		vtk_radiobutton.pack(side=Tkinter.LEFT)
		mat_radiobutton = Tkinter.Radiobutton(format_frame, text=".mat", variable=self.file_format, value='.mat')
		mat_radiobutton.pack(side=Tkinter.LEFT)
		
		# Start button
		button_run = Tkinter.Button(self.root, text ="Start EZyRB", command = self._start_ezyrb, bg='#065893', fg='#f19625', font='bold')
		button_run.pack()
		self.label_new_mu = Tkinter.Label(self.root, text='Start EZyRB to find the new parameter value')
		self.label_new_mu.pack()
		self.label_error= Tkinter.Label(self.root, text='Start EZyRB to find the maximum error')
		self.label_error.pack()
		
		# Enrich database button
		button_run = Tkinter.Button(self.root, text ="Enrich", command = self._add_snapshot, bg='green', fg='white', font='bold')
		button_run.pack()
		
		# Enrich database button
		button_finish = Tkinter.Button(self.root, text ="Finish", command = self._finish, bg='red', fg='white', font='bold')
		button_finish.pack()

		# Menu
		menubar = Tkinter.Menu(self.root)
		helpmenu = Tkinter.Menu(menubar, tearoff=0)
		helpmenu.add_command(label="About...", command=self._goto_website)
		menubar.add_cascade(label="Help", menu=helpmenu)
		self.root.config(menu=menubar)


	def start(self):
	
		self.root.mainloop()

