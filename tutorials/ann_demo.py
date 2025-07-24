import numpy as np
import matplotlib.pyplot as plt
import torch
from ezyrb import Approximation
from ezyrb.approximation import ANN

class ANN_Demo(ANN):
    def __init__(self, mrom, layers, function, stop_training, loss=None,
                 optimizer=torch.optim.Adam, lr=0.001, l2_regularization=0,
                 frequency_print=500, last_identity=True):
        super().__init__(layers, function, stop_training, loss=None,
                 optimizer=torch.optim.Adam, lr=0.001, l2_regularization=0,
                 frequency_print=10, last_identity=True)

        # import useful data from multirom and roms predictions
        self.mrom = mrom
        self.params = list(self.mrom.roms.values())[0].validation_full_database.parameters_matrix
        
        self.frequency_print = frequency_print
        self.lr = lr
        self.l2_regularization = l2_regularization

        # import ROMs and validation predictions of all ROMs
        self.rom_validation_predictions = {}
        for rom in self.mrom.roms:
            rom_pred = self.mrom.roms[rom]
            rom_pred = rom_pred.predict(self.params)
            rom_pred = rom_pred.reshape(rom_pred.shape[0]*rom_pred.shape[1], 1)
            self.rom_validation_predictions[rom] = self._convert_numpy_to_torch(rom_pred)
            
        # Device configuration
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: 💻 {self.device}")
            
    def init_weights(self, model):
        '''
        Initialize the weights of the ANN model using He initialization.
        '''
        
        if isinstance(model, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(model.weight, nonlinearity='relu') # He initialization
            if model.bias is not None:
                torch.nn.init.constant_(model.bias, 0.1)

    def _build_model_(self, points):
        layers = self.layers.copy()
        layers.insert(0, points.shape[1])
        layers.append(len(self.mrom.roms))
        self.model = self._list_to_sequential(layers, self.function)
        
        # Initialize the weights of the ANN
        self.model.apply(self.init_weights)
        
        # Move the model to the device
        self.model.to(self.device)

    def fit(self, points, values): # points=(x, mu) and values=(snapshots)
        self._build_model_(points)
        optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.l2_regularization)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1000)

        points = self._convert_numpy_to_torch(points)
        values = self._convert_numpy_to_torch(values)
        
        # Move everything to the device
        points = points.to(self.device)
        values = values.to(self.device)
        self.rom_validation_predictions = {rom: pred.to(self.device) for rom, pred in self.rom_validation_predictions.items()}

        # train the neural network
        n_epoch = 1
        flag = True
        while flag:
            # compute output of ANN
            y_pred = self.model(points)

            # compute aggregated solution from output weights of ANN
            aggr_pred = torch.zeros(values.shape, device=self.device)
            for i, rom in enumerate(self.mrom.roms):
                weight = y_pred.clone()[..., i].unsqueeze(-1)
                aggr_pred += weight*self.rom_validation_predictions[rom]

            # difference between aggregated solution and exact solution
            loss = self.loss(aggr_pred, values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scalar_loss = loss.item()
            self.loss_trend.append(scalar_loss)
            
            scheduler.step(scalar_loss)

            for criteria in self.stop_training:
                if isinstance(criteria, int):  # stop criteria is an integer
                    if n_epoch == criteria:
                        flag = False
                elif isinstance(criteria, float):  # stop criteria is float
                    if scalar_loss < criteria:
                        flag = False

            if (flag is False or
                    n_epoch == 1 or n_epoch % self.frequency_print == 0):
                print(f'[epoch {n_epoch:6d}]\t{scalar_loss:e}')

            n_epoch += 1

        return optimizer

    def predict(self, x):
        
        # Move the model to the device
        x = self._convert_numpy_to_torch(np.array(x))
        x = x.to(self.device)
        y_new = self.model(x)
        ynew = y_new.cpu().detach().numpy()
        return ynew
    
    def plot_loss(self):
        '''
        Plot the loss function for visualization.
        '''
        fig = plt.figure(figsize=(10, 6))
        plt.plot(self.loss_trend, marker='o', color='blue', markersize=3)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Train Loss', fontsize=12)
        plt.yscale('log')
        plt.title('Train Loss vs Epochs', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        
        plt.show()
        
    def save(self, filename):
        '''
        Save the model to a file.
        '''
        torch.save(self.model.state_dict(), filename)
        
    def load(self, filename):
        '''
        Load the model from a file.
        '''
        self.model.load_state_dict(torch.load(filename))
        self.model.to(self.device)
