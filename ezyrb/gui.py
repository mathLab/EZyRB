"""
Utilities for handling the Graphic Unit Interface.

.. todo::
    - Switch to Ttk instead of Tk for a better look of the GUI.
    - Insert a button to decide if plot or not the singular values.
    - Insert a button to decide the path where to save the structures at the end of the procedure.
    - Use grid instead of pack
        
"""

try:
    import tkinter
    from tkinter.filedialog import askopenfilename
except:
    import Tkinter as tkinter
    from tkFileDialog import askopenfilename

import numpy as np
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
    :cvar string url: url of the github page of EZyRB.
    :cvar string tria_path: path of the triangulation file.
    
    :cvar string tria_path: path of the pod_basis file.
    
    :cvar string parsing_file_path: path of the file to be parsed in order to allow to write the new output.
    :cvar string new_mu: new parameter values. The values must be separated by a comma.
    :cvar string outfilename: name of the new output file.
    :cvar string finish_label: string that says when the online step is done.
    :cvar tkinter.Label label_new_mu: label where to print the new parameter value.
    :cvar tkinter.Label label_error: label where to print the maximum error on the tesselation.
    :cvar tkinter.Label label_tria: label where to print the triangulation file path.
    
    :cvar tkinter.Label label_basis: label where to print the pod basis file path.
    
    :cvar tkinter.Label label_parsing_file: label where to print the parsing file path.
    :cvar tkinter.Label label_finish_online: label where to print the finish message.
    :cvar Pod/Interp ezyrb_handler: class for the model reduction. It can be both a Pod and a 
        Interp class (it depends on the is_scalar_switch boolean).
    :cvar bool is_scalar_switch: switch to decide is the output of interest is a scalar or a field.
        
    """

    def __init__(self):
        self.root = tkinter.Tk()
        self.root.title('EZyRB')
        self.output_name = tkinter.StringVar()
        self.weights_name = tkinter.StringVar()
        self.namefile_prefix = tkinter.StringVar()
        self.file_format = tkinter.StringVar()
        self.url = 'https://github.com/mathLab/EZyRB'
        self.tria_path = tkinter.StringVar()
        self.basis_path = tkinter.StringVar()
        self.parsing_file_path = tkinter.StringVar()
        self.new_mu = tkinter.StringVar()
        self.outfilename = tkinter.StringVar()
        self.finish_label = tkinter.StringVar()
        self.label_new_mu = None
        self.label_error = None
        self.label_tria = None
        self.label_basis = None
        self.label_parsing_file = None
        self.label_finish_online = None
        self.ezyrb_handler = None
        self.is_scalar_switch = tkinter.BooleanVar()
        self.logo_label = None
        self.img = None

    def _start_ezyrb_offline(self):
        """
        The private method starts the ezyrb algorithm. Offline Step.
        """
        '''
        #output_name = 'Pressure_drop'
        output_name = 'Pressure'
        weights_name = 'Weights'
        #namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
        namefile_prefix = 'tests/test_datasets/matlab_0'
        file_format = '.vtk
        '''
        import ezyrb as ez
        if self.is_scalar_switch.get() != True:
            #self.ezyrb_handler = ez.pod.Pod(output_name, weights_name, namefile_prefix, file_format)
            self.ezyrb_handler = ez.pod.Pod(self.output_name.get(),
                                            self.weights_name.get(),
                                            self.namefile_prefix.get(),
                                            self.file_format.get())
        else:
            #self.ezyrb_handler = ez.interpolation.Interp(output_name, namefile_prefix, file_format)
            self.ezyrb_handler = ez.interpolation.Interp(
                self.output_name.get(), self.namefile_prefix.get(),
                self.file_format.get())

        self.ezyrb_handler.start()
        self.label_new_mu.configure(text='New parameter value ' + str(
            self.ezyrb_handler.cvt_handler.mu_values[:, -1]))
        self.label_error.configure(
            text='Error ' + str(self.ezyrb_handler.cvt_handler.max_error))

    def _start_ezyrb_online(self):
        """
        The private method starts the ezyrb algorithm. Online Step.
        """
        import ezyrb as ez
        mu_value = np.fromstring(self.new_mu.get(), dtype=float, sep=',')
        directory = (os.path.dirname(self.basis_path.get()) + '/')

        online_handler = ez.online.Online(
            mu_value,
            self.output_name.get(),
            directory=directory,
            is_scalar=self.is_scalar_switch.get())
        online_handler.run()
        online_handler.write_file(self.outfilename.get(),
                                  self.parsing_file_path.get())

        self.finish_label.set('Online step ended. New output file saved.')

    def _chose_tria_file(self):
        """
        The private method explores the file system and allows to select the wanted triangulation file.
        Up to now, you can select only .npy file.
        """
        filename_tria = askopenfilename(filetypes=[("Python file", ('*.npy'))])
        self.tria_path.set(filename_tria)
        self.label_tria.configure(fg='green')

    def _chose_basis_file(self):
        """
        The private method explores the file system and allows to select the wanted triangulation file.
        Up to now, you can select only .npy file.
        """
        filename_basis = askopenfilename(filetypes=[("Python file", ('*.npy'))])
        self.basis_path.set(filename_basis)
        self.label_basis.configure(fg='green')

    def _chose_parsing_file(self):
        """
        The private method explores the file system and allows to select the wanted triangulation file.
        Up to now, you can select only .npy file.
        """
        filename_parsing_file = askopenfilename(filetypes=[(
            "VTK file", '*.vtk'), ("Matlab file", '*.mat'), ('All', '*')])
        self.parsing_file_path.set(filename_parsing_file)
        self.label_parsing_file.configure(fg='green')

    def _add_snapshot(self):
        """
        The private method adds a snapshot to the database.
        """
        self.ezyrb_handler.add_snapshot()
        self.label_new_mu.configure(text='New parameter value' + str(
            self.ezyrb_handler.cvt_handler.mu_values[:, -1]))
        self.label_error.configure(
            text='Error ' + str(self.ezyrb_handler.cvt_handler.max_error))

    def _finish(self):
        """
        The private method to stop the iterations and save the structures necessary for the online step.
        """
        self.ezyrb_handler.write_structures()

    def _quit(self):
        """
        The private method close the program.
        """
        self.root.destroy()

    def _goto_website(self):
        """
        The private method opens the EZyRB main page on github. 
        It is used for info about EZyRB in the menu.
        """
        webbrowser.open(self.url)

    def _main(self):
        """
        The private method inizializes and visualizes the window.
        """
        self.logo_panel = tkinter.Canvas(self.root, height=60, width=60)
        self.logo_panel.pack(side="bottom", padx=5, pady=5, anchor=tkinter.SE)
        try:
            self.img = tkinter.PhotoImage(
                master=self.logo_panel, file='readme/logo_EZyRB_gui.gif')
        except:
            self.img = tkinter.PhotoImage(
                master=self.logo_panel, file='../readme/logo_EZyRB_gui.gif')
        self.logo_panel.create_image(35, 35, image=self.img)

        online_offline_frame = tkinter.Frame(self.root)
        online_offline_frame.pack()

        ## OFFLINE
        offline_frame = tkinter.Frame(
            online_offline_frame,
            relief=tkinter.GROOVE,
            borderwidth=1,
            bg='#c1d0f0')
        offline_frame.grid(row=0, column=0, padx=5, pady=5)
        tkinter.Label(
            offline_frame, text="OFFLINE", bg='#c1d0f0', font=("Arial",
                                                               20)).pack()

        text_input_frame = tkinter.Frame(
            offline_frame, relief=tkinter.GROOVE, borderwidth=1)
        text_input_frame.pack(padx=5, pady=5, anchor=tkinter.W)

        # Buttons 1
        tkinter.Label(
            text_input_frame, text="Path and prefix").grid(
                row=0, column=0)
        tkinter.Entry(
            text_input_frame, bd=5, textvariable=self.namefile_prefix).grid(
                row=0, column=1)

        # Button 2
        tkinter.Label(
            text_input_frame, text="Output of interest").grid(
                row=2, column=0)
        tkinter.Entry(
            text_input_frame, bd=5, textvariable=self.output_name).grid(
                row=2, column=1)

        tkinter.Label(
            text_input_frame, text="Output is a").grid(
                row=3, column=0, pady=2)
        switch_frame = tkinter.Frame(text_input_frame)
        switch_frame.grid(row=3, column=1, pady=2)
        tkinter.Radiobutton(
            switch_frame,
            text="Scalar",
            variable=self.is_scalar_switch,
            value=True).pack(side=tkinter.LEFT)
        tkinter.Radiobutton(
            switch_frame,
            text="Field",
            variable=self.is_scalar_switch,
            value=False).pack(side=tkinter.RIGHT)

        # Button 3
        label_weights = tkinter.Label(
            text_input_frame, text="Weight name").grid(
                row=1, column=0)
        tkinter.Entry(
            text_input_frame, bd=5, textvariable=self.weights_name).grid(
                row=1, column=1)

        # Button 4
        format_frame = tkinter.Frame(text_input_frame)
        format_frame.grid(row=4, column=1, pady=2)

        tkinter.Label(
            text_input_frame, text="Select file format").grid(
                row=4, column=0, pady=2)

        tkinter.Radiobutton(
            format_frame, text=".vtk", variable=self.file_format,
            value='.vtk').pack(side=tkinter.LEFT)
        tkinter.Radiobutton(
            format_frame, text=".mat", variable=self.file_format,
            value='.mat').pack(side=tkinter.RIGHT)

        # Start button
        start_frame_offline = tkinter.Frame(offline_frame)
        start_frame_offline.pack(padx=10, pady=10)
        tkinter.Button(
            start_frame_offline,
            text="Start EZyRB",
            command=self._start_ezyrb_offline,
            bg='#065893',
            fg='#f19625',
            font='bold').pack(
                padx=5, pady=5)

        display_frame = tkinter.Frame(
            offline_frame, relief=tkinter.GROOVE, borderwidth=1)
        display_frame.pack()

        self.label_new_mu = tkinter.Label(
            display_frame, text='Start EZyRB to find the new parameter value')
        self.label_new_mu.pack(padx=0, pady=2, anchor=tkinter.W)
        self.label_error = tkinter.Label(
            display_frame, text='Start EZyRB to find the maximum error')
        self.label_error.pack(padx=0, pady=0, anchor=tkinter.W)

        # Enrich database button
        chose_frame = tkinter.Frame(offline_frame)
        chose_frame.pack(padx=5, pady=5)
        tkinter.Button(
            chose_frame,
            text="Enrich",
            command=self._add_snapshot,
            bg='green',
            fg='white',
            font='bold').pack(
                side=tkinter.LEFT, padx=5, pady=5)

        # Finish button
        tkinter.Button(
            chose_frame,
            text="Finish",
            command=self._finish,
            bg='red',
            fg='white',
            font='bold').pack(
                side=tkinter.RIGHT, padx=5, pady=5)

        ## ONLINE
        online_frame = tkinter.Frame(
            online_offline_frame,
            relief=tkinter.GROOVE,
            borderwidth=1,
            bg='#80ff80')
        online_frame.grid(row=0, column=1, padx=5, pady=5)
        tkinter.Label(
            online_frame, text="ONLINE", bg='#80ff80', font=("Arial",
                                                             20)).pack()

        text_input_online_frame = tkinter.Frame(
            online_frame, relief=tkinter.GROOVE, borderwidth=1)
        text_input_online_frame.pack(padx=5, pady=5, anchor=tkinter.W)

        tkinter.Button(
            text_input_online_frame,
            text="Pick triangulation",
            command=self._chose_tria_file).grid(
                row=0, column=0)
        self.label_tria = tkinter.Label(
            text_input_online_frame, textvariable=self.tria_path, fg='red')
        self.tria_path.set("No triangulation chosen!")
        self.label_tria.grid(row=0, column=1)

        tkinter.Button(
            text_input_online_frame,
            text="Pick pod basis",
            command=self._chose_basis_file).grid(
                row=1, column=0)
        self.label_basis = tkinter.Label(
            text_input_online_frame, textvariable=self.basis_path, fg='red')
        self.basis_path.set("No basis chosen!")
        self.label_basis.grid(row=1, column=1)

        tkinter.Label(
            text_input_online_frame, text="Output of interest").grid(
                row=2, column=0)
        tkinter.Entry(
            text_input_online_frame, bd=5, textvariable=self.output_name).grid(
                row=2, column=1)

        tkinter.Label(
            text_input_online_frame, text="Output is a").grid(
                row=3, column=0, pady=2)
        switch_frame_online = tkinter.Frame(text_input_online_frame)
        switch_frame_online.grid(row=3, column=1, pady=2)
        tkinter.Radiobutton(
            switch_frame_online,
            text="Scalar",
            variable=self.is_scalar_switch,
            value=True).pack(side=tkinter.LEFT)
        tkinter.Radiobutton(
            switch_frame_online,
            text="Field",
            variable=self.is_scalar_switch,
            value=False).pack(side=tkinter.RIGHT)

        format_frame_online = tkinter.Frame(text_input_online_frame)
        format_frame_online.grid(row=4, column=1, pady=2)

        tkinter.Label(
            text_input_online_frame, text="Select file format").grid(
                row=4, column=0, pady=2)
        tkinter.Radiobutton(
            format_frame_online,
            text=".vtk",
            variable=self.file_format,
            value='.vtk').pack(side=tkinter.LEFT)
        tkinter.Radiobutton(
            format_frame_online,
            text=".mat",
            variable=self.file_format,
            value='.mat').pack(side=tkinter.RIGHT)

        tkinter.Button(
            text_input_online_frame,
            text="File for parsing",
            command=self._chose_parsing_file).grid(
                row=5, column=0)
        self.label_parsing_file = tkinter.Label(
            text_input_online_frame,
            textvariable=self.parsing_file_path,
            fg='red')
        self.parsing_file_path.set("No parsing file chosen!")
        self.label_parsing_file.grid(row=5, column=1)

        tkinter.Label(
            text_input_online_frame, text="New parameter").grid(
                row=6, column=0)
        tkinter.Entry(
            text_input_online_frame, bd=5, textvariable=self.new_mu).grid(
                row=6, column=1)

        tkinter.Label(
            text_input_online_frame, text="Output file name").grid(
                row=7, column=0)
        tkinter.Entry(
            text_input_online_frame, bd=5, textvariable=self.outfilename).grid(
                row=7, column=1)

        start_frame_online = tkinter.Frame(online_frame)
        start_frame_online.pack(padx=10, pady=10)
        tkinter.Button(
            start_frame_online,
            text="Start EZyRB",
            command=self._start_ezyrb_online,
            bg='#065893',
            fg='#f19625',
            font='bold').pack(
                padx=5, pady=5)

        self.label_finish_online = tkinter.Label(
            online_frame, textvariable=self.finish_label, bg='#80ff80')
        self.finish_label.set("")
        self.label_finish_online.pack()

        # Menu
        menubar = tkinter.Menu(self.root)

        mainmenu = tkinter.Menu(menubar, tearoff=0)
        mainmenu.add_command(label="Quit", command=self._quit)
        menubar.add_cascade(label="EZyRB", menu=mainmenu)

        helpmenu = tkinter.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About...", command=self._goto_website)
        menubar.add_cascade(label="Help", menu=helpmenu)

        self.root.config(menu=menubar)

    def start(self):
        """
        This method inizializes and starts the GUI. 
        """
        self._main()
        self.root.mainloop()


if __name__ == "__main__":
    app = Gui()
    app.start()
