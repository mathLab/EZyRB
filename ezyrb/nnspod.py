import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import torch.nn as nn
from .ann import ANN
from .pod import POD
from .database import Database


class NNsPOD(POD):
    def __init__(self, method = "svd", path = None):
        ## add loss, layers, and functions variables
        super().__init__(method)
        self.path = path


    def reshape2dto1d(self, x, y):
        """
        reshapes two n by n arrays into one n^2 by 2 array
        :param numpy.array x: x value of data
        :param numpy.array y: y value of data
        """
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        coords = np.concatenate((x, y), axis = 1)
        coords = np.array(coords).reshape(-1,2)

        return coords

    def reshape1dto2d(self, snapshots):
        """
        turns 1d list of data into 2d
        :param array-like snapshots: data to be reshaped
        """
        return snapshots.reshape(int(np.sqrt(len(snapshots))), int(np.sqrt(len(snapshots))))

    def train_interpnet(self,ref_data, interp_layers, interp_function, interp_stop_training, interp_loss, retrain = False, frequency_print = 0, save = True):
        """
        trains the Interpnet given 1d data:

        :param database ref_data: the reference data that the rest of the data will be shifted to
        :param list interp_layers: list with number of neurons in each layer
        :param torch.nn.modules.activation interp_function: activation function for the interpnet
        :param float interp_stop_training: desired tolerance for the interp training
        :param torch.nn.Module interp_loss: loss function (MSE default)
        :param boolean retrain: True if the interpNetShould be retrained, False if it should be loaded
        """

        self.interp_net = ANN(interp_layers, interp_function, interp_stop_training, interp_loss)
        if len(ref_data.space.shape) > 2:
            space = ref_data.space.reshape(-1, 2)
        else:
            space = ref_data.space.reshape(-1,1)
        snapshots = ref_data.snapshots.reshape(-1,1)
        if not retrain:
            try:
                self.interp_net = self.interp_net.load_state(self.path, space, snapshots)
                print("loaded interpnet")
            except:
                self.interp_net.fit(space, snapshots, frequency_print = frequency_print)
                if save:
                    self.interp_net.save_state(self.path)
        else:
            self.interp_net.fit(space, snapshots, frequency_print = frequency_print)
            if save:
                self.interp_net.save_state(self.path)

    def shift(self, x, y, shift_quantity):
        """
        shifts data by shift_quanity
        """
        return(x+shift_quantity, y)

    def pre_shift(self,x,y, ref_y):
        """
        moves data so that the max of y and max of ref_y are at the same x coordinate
        """
        maxy = 0
        for i, n, in enumerate(y):
            if n > y[maxy]:
                maxy = i
        maxref = 0
        for i, n in enumerate(ref_y):
            if n > ref_y[maxref]:
                maxref = i

        return self.shift(x, y, x[maxref]-x[maxy])[0]

    def make_points(self, x, params):
        """
        creates points that can be used to train and predict shiftnet
        """
        if len(x.shape)> 1:
            points = np.zeros((len(x),3))
            for j, s in enumerate(x):
                points[j][0] = s[0]
                points[j][1] = s[1]
                points[j][2] = params[0]
        else:
            points = np.zeros((len(x),2))
            for j, s in enumerate(x):
                points[j][0] = s
                points[j][1] = params[0]
        return points

    def build_model(self, dim = 1):
        """
        builds model based on dimension of input data
        """
        layers = self.layers.copy()
        layers.insert(0, dim + 1)
        layers.append(dim)
        layers_torch = []
        for i in range(len(layers) - 2):
            layers_torch.append(nn.Linear(layers[i], layers[i + 1]))
            layers_torch.append(self.function)
        layers_torch.append(nn.Linear(layers[-2], layers[-1]))
        self.model = nn.Sequential(*layers_torch)

    def train_shiftnet(self, db, shift_layers, shift_function, shift_stop_training, 
                        ref_data, preshift = False, 
                        optimizer = torch.optim.Adam, learning_rate = 0.0001, frequency_print = 0):
        """
        Trains and evaluates shiftnet given 1d data 'db'

        :param Database db: data at a certain parameter value
        :param list shift_layers: ordered list with number of neurons in each layer
        :param torch.nn.modeulse.activation shift_function: the activation function used by the shiftnet
        :param int, float, or list stop_training: 
            int: number of epochs before stopping
            float: desired tolarance before stopping training
            list: a int and a float, stops when either desired epochs or tolerance is reached
        :param Database db: data at the reference datapoint
        :param boolean preshift: True if preshift is desired otherwise false.
        """
        self.layers = shift_layers
        self.function = shift_function
        self.loss_trend = []
        if preshift:
            x = self.pre_shift(db.space[0], db.snapshots[0], ref_data.snapshots[0])
        else:
            x = db.space[0]
        if len(db.space.shape) > 2:
            x_reshaped = x.reshape(-1,2)
            self.build_model(dim = 2)
        else:
            self.build_model(dim = 1)
            x_reshaped = x.reshape(-1,1)

        values = db.snapshots.reshape(-1,1)

        self.stop_training = shift_stop_training
        points = self.make_points(x, db.parameters)

        self.optimizer = optimizer(self.model.parameters(), lr = learning_rate)

        self.loss = torch.nn.MSELoss()
        points = torch.from_numpy(points).float()
        n_epoch = 1
        flag = True
        while flag:
            shift = self.model(points)
            x_shift, y = self.shift(
                        torch.from_numpy(x_reshaped).float(),
                        torch.from_numpy(values).float(),
                        shift)
            ref_interp = self.interp_net.model(x_shift)
            loss = self.loss(ref_interp, y)
            loss.backward()
            self.optimizer.step()
            self.loss_trend.append(loss.item())
            for criteria in self.stop_training:
                if isinstance(criteria, int):  # stop criteria is an integer
                    if n_epoch == criteria:
                        flag = False
                elif isinstance(criteria, float):  # stop criteria is float
                    if loss.item() < criteria:
                        flag = False
            if frequency_print != 0:
                if n_epoch % frequency_print == 1:
                    print(loss.item())
            n_epoch += 1

        new_point = self.make_points(x, db.parameters)
        shift = self.model(torch.from_numpy(new_point).float())
        x_new = self.shift(
                        torch.from_numpy(x_reshaped).float(),
                        torch.from_numpy(values).float(),
                        shift)[0]
        x_ret = x_new.detach().numpy()
        return x_ret
    
    def fit(self, db, ref_point, interp_loss, interp_function, interp_layers,
            shift_loss, shift_function,shift_layers):
        ## input variables: load files.
        self.train_interpnet(db[ref_point], interp_layers, interp_function, interp_loss, None, retrain  = False, frequency_print = 25)
        new_x = np.zeros(shape = db.space.shape)
        i = 0
        while i < db.parameters.shape[0]:
            if len(db.space.shape) > 2:
                new_x[i] = self.train_shiftnet(db[i], shift_layers, shift_function, shift_loss, db[ref_point], preshift = True, frequency_print = 50).reshape(-1, 2)
            else:
                new_x[i] = self.train_shiftnet(db[i], shift_layers, shift_function, shift_loss, db[ref_point], preshift = True, frequency_print = 50).reshape(-1)
            i+=1
            if i == ref_point:
                new_x[ref_point] = db.space[ref_point]
                i +=1
        db = Database(space = new_x, snapshots = db.snapshots, parameters = db.parameters)

        i = 0
        new_snapshots = np.zeros(shape = db.snapshots.shape)
        new_space = np.zeros(shape = db.space.shape)
        while i < db.parameters.shape[0]:
            self.train_interpnet(db[i], interp_layers, interp_function, interp_loss, None, retrain  = True, frequency_print = 200, save = False)
            new_snapshots[i] = self.interp_net.model(torch.from_numpy(db.space[ref_point].reshape(-1,1)).float()).detach().numpy().reshape(-1)
            new_space[i] = db.space[ref_point]
            i+=1
            if i == ref_point:
                new_snapshots[ref_point] = db.snapshots[ref_point]
                new_space[ref_point] =  db.space[ref_point]
                i +=1

        db = Database(space = new_space, snapshots = new_snapshots, parameters = db.parameters)
        POD_ = POD(method = 'svd')
        return POD_.fit(db.snapshots)

