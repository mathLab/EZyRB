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
        super().__init__(method)
        self.path = path
    

    def reshape2dto1d(self, x, y):
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        coords = np.concatenate((x, y), axis = 1)
        coords = np.array(coords).reshape(-1,2)
        
        
        return coords

    def reshape1dto2d(self, snapshots):
        print(len(snapshots), snapshots.shape)
        return snapshots.reshape(int(np.sqrt(len(snapshots))), int(np.sqrt(len(snapshots))))


    def train_InterpNet1d(self,ref_data, interp_layers, interp_function, interp_stop_training, interp_loss, retrain = False):

        # print("loading")
        
        self.interp_net = ANN(interp_layers, interp_function, interp_stop_training, interp_loss)
        if not retrain:
            try:
                self.interp_net = self.interp_net.load_state(self.path, ref_data.space.reshape(-1,1), ref_data.snapshots.reshape(-1,1))
                print("loaded")
            except:
                self.interp_net.fit(ref_data.space.reshape(-1,1), ref_data.snapshots.reshape(-1,1))
                self.interp_net.save_state(self.path)
                print(self.interp_net.load_state(self.path, ref_data.space.reshape(-1,1), ref_data.snapshots.reshape(-1,1)))
        else:
            self.interp_net.fit(ref_data.space.reshape(-1,1), ref_data.snapshots.reshape(-1,1))
            self.interp_net.save_state(self.path)
        #plt.plot(ref_data.space, ref_data.snapshots, "o")
        xi = np.linspace(0,5,1000).reshape(-1,1)
        yi = self.interp_net.predict(xi)
        print(xi.shape, yi.shape)
        #plt.plot(xi,yi, ".")
        #plt.show()


    def train_InterpNet2d(self,ref_data, interp_layers, interp_function, interp_stop_training, interp_loss, retrain = False):

        
        self.interp_net = ANN(interp_layers, interp_function, interp_stop_training, interp_loss)
        space = ref_data.space.reshape(-1, 2)
        snapshots = ref_data.snapshots.reshape(-1, 1)
        
        if not retrain:
            try:
                self.interp_net = self.interp_net.load_state(self.path, space, snapshots)
            except:
                self.interp_net.fit(space, snapshots)
                self.interp_net.save_state(self.path)
        else:
            self.interp_net.fit(space, snapshots)
            self.interp_net.save_state(self.path)

        x = np.linspace(0, 5, 256)
        y = np.linspace(0, 5, 256)
        gridx, gridy = np.meshgrid(x, y)
        
        plt.pcolor(gridx,gridy,ref_data.snapshots.reshape(256, 256))
        plt.show()
        res = 1000
        x = np.linspace(0, 5, res)
        y = np.linspace(0, 5, res)
        gridx, gridy = np.meshgrid(x, y)
        input = self.reshape2dto1d(gridx, gridy)
        output = self.interp_net.predict(input)

        toshow = self.reshape1dto2d(output)
        plt.pcolor(gridx,gridy,toshow)
        plt.show()
        
    


    def shift(self, x, y, shift_quantity):
        return(x+shift_quantity, y)
    
    def pre_shift(self,x,y, ref_y):
        maxy = 0
        for i, n, in enumerate(y):
            if n > y[maxy]:
                maxy = i
        maxref = 0
        for i, n in enumerate(ref_y):
            if n > ref_y[maxref]:
                maxref = i
        
        print( x[maxref]-x[maxy], maxref, maxy)
        return self.shift(x, y, x[maxref]-x[maxy])[0]
    
    def make_points(self, x, params):
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
        layers = self.layers.copy()
        layers.insert(0, dim + 1)
        print(layers, "!!!!")
        layers.append(dim)
        layers_torch = []
        for i in range(len(layers) - 2):
            layers_torch.append(nn.Linear(layers[i], layers[i + 1]))
            layers_torch.append(self.function)
        layers_torch.append(nn.Linear(layers[-2], layers[-1]))
        self.model = nn.Sequential(*layers_torch)



    def train_ShiftNet1d(self, db, shift_layers, shift_function, shift_stop_training, ref_data, preshift = False):
        # TODO: 
        # make sure neural net works no mater distance between data
        # check and implement 2d functionality
        # make code look better
        self.layers = shift_layers
        self.function = shift_function
        self.loss_trend = []
        if preshift:
            x = self.pre_shift(db.space[0], db.snapshots[0], ref_data.snapshots[0])
        else:
            x = db.space[0]
        
        self.stop_training = shift_stop_training
        points = self.make_points(x, db.parameters)
        values = db.snapshots.reshape(-1,1)
        self.build_model(dim = 1)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 0.0001)

        self.loss = torch.nn.MSELoss()
        points = torch.from_numpy(points).float()
        n_epoch = 1
        flag = True
        while flag:
            shift = self.model(points)
            x_shift, y = self.shift(
                        torch.from_numpy(x.reshape(-1,1)).float(),
                        torch.from_numpy(db.snapshots.reshape(-1,1)).float(),
                        shift)
            #print(x_shift,y)
            ref_interp = self.interp_net.predict_tensor(x_shift)
            #print(ref_interp)
            loss = self.loss(ref_interp, y)
            print(loss.item())
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
            n_epoch += 1

        new_point = self.make_points(x, db.parameters)
        shift = self.model(torch.from_numpy(new_point).float())
        x_new = self.shift(
                        torch.from_numpy(x.reshape(-1,1)).float(),
                        torch.from_numpy(db.snapshots.reshape(-1,1)).float(),
                        shift)[0]
        
        plt.plot(db.space, db.snapshots, "go")
        plt.plot(x_new.detach().numpy(), db.snapshots.reshape(-1,1), ".")
        return shift

    def train_ShiftNet2d(self, db, shift_layers, shift_function, shift_stop_training, ref_data, preshift = False):
        # TODO: 
        # make sure neural net works no mater distance between data
        # check and implement 2d functionality
        # make code look better
        # work on pre_shift for 2d data (iterate through all data until max is found)
        # make sure shift works for 2d data(might only shift one part)
        self.layers = shift_layers
        self.function = shift_function
        self.loss_trend = []
        if preshift:
            x = self.pre_shift(db.space[0], db.snapshots[0], ref_data.snapshots[0])
        else:
            x = db.space[0]
        
        self.stop_training = shift_stop_training
        points = self.make_points(x, db.parameters)
        self.build_model(dim = 2)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 0.00001)

        self.loss = torch.nn.MSELoss()
        points = torch.from_numpy(points).float()
        n_epoch = 1
        flag = True
        while flag:
            shift = self.model(points)
            x_shift, y = self.shift(
                        torch.from_numpy(x.reshape(-1,2)).float(),
                        torch.from_numpy(db.snapshots.reshape(-1,1)).float(),
                        shift)
            #print(x_shift,y)
            ref_interp = self.interp_net.predict_tensor(x_shift)
            #print(ref_interp)
            loss = self.loss(ref_interp, y)
            print(loss.item())
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
            n_epoch += 1
    
            
        x = np.linspace(0, 5, 256)
        y = np.linspace(0, 5, 256)
        gridx, gridy = np.meshgrid(x, y)
        
        plt.pcolor(gridx,gridy,ref_data.snapshots.reshape(256, 256))
        plt.show()
        res = 256
        x = np.linspace(0, 5, res)
        y = np.linspace(0, 5, res)
        gridx, gridy = np.meshgrid(x, y)
        coords = self.reshape2dto1d(gridx, gridy)
        new_point = self.make_points(coords, db.parameters)        
        shift = self.model(torch.from_numpy(new_point).float())
        x_new = self.shift(
                        torch.from_numpy(coords.reshape(-1,2)).float(),
                        torch.from_numpy(db.snapshots.reshape(-1,1)).float(),
                        shift)[0]
        print(x_new.shape)
        x, y = np.hsplit(x_new.detach().numpy(), 2)
        x = self.reshape1dto2d(x)
        y = self.reshape1dto2d(y)
        snapshots = self.reshape1dto2d(db.snapshots.reshape(-1,1))
        print(x.shape, y.shape)
        plt.pcolor(x,y,snapshots)
        plt.show()
        return shift