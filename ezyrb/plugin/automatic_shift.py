""" Module for Scaler plugin """

import numpy as np
import torch

from .plugin import Plugin


class AutomaticShiftSnapshots(Plugin):
    """
    The plugin implements the automatic "shifting" preprocessing: exploiting a
    machine learning framework, it is able to detect the quantity to shift the
    snapshots composing the database, such that the reduction method performs
    better, depending on the problem at hand.

    Reference: Papapicco, D., Demo, N., Girfoglio, M., Stabile, G., & Rozza, G.
    (2022). The Neural Network shifted-proper orthogonal decomposition: A
    machine learning approach for non-linear reduction of hyperbolic equations.
    Computer Methods in Applied Mechanics and Engineering, 392, 114687.

    :param callable shift_function: a user defined function that return the
        shifting quantity for any the snapshot, given the corresponding input
        parameter.
    :param Approximation interpolator: the interpolator to use to evaluate the
        shifted snapshots on some reference space.
    :param int parameter_index: in case of multi-dimensional parameter,
        indicate the index of the parameter component to pass to the shift
        function. Default is 0.
    :param int reference_index: indicate the index of the snapshots within the
        database whose space will be used as reference space. Default is 0.

    Example:

    >>> from ezyrb import POD, RBF, Database, Snapshot, Parameter, Linear, ANN
    >>> from ezyrb import ReducedOrderModel as ROM
    >>> from ezyrb.plugin import AutomaticShiftSnapshots
    >>> interp = ANN([10, 10], torch.nn.Softplus(), 1000, frequency_print=50, lr=0.03)
    >>> shift = ANN([], torch.nn.LeakyReLU(), [2000, 1e-3], frequency_print=50, l2_regularization=0,  lr=0.002)
    >>> nnspod = AutomaticShiftSnapshots(shift, interp, Linear(fill_value=0.0), barycenter_loss=10.)
    >>> pod = POD(rank=1)
    >>> rbf = RBF()
    >>> db = Database()
    >>> for param in params:
    >>>     space, values = wave(param)
    >>>     snap = Snapshot(values=values, space=space)
    >>>     db.add(Parameter(param), snap)
    >>> rom = ROM(db, pod, rbf, plugins=[nnspod])
    >>> rom.fit()
    """
    def __init__(self, shift_network, interp_network, interpolator,
                 parameter_index=0, reference_index=0, barycenter_loss=0):
        super().__init__()

        self.interpolator = interpolator
        self.shift_network = shift_network
        self.interp_network = interp_network
        self.parameter_index = parameter_index
        self.reference_index = reference_index
        self.barycenter_loss = barycenter_loss

    def _train_interp_network(self):
        """
        """
        self.interp_network.fit(
            self.reference_snapshot.space.reshape(-1, 1),
            self.reference_snapshot.values.reshape(-1, 1)
        )

    def _train_shift_network(self, db):
        """
        """
        ref_center = torch.tensor(np.average(
            self.reference_snapshot.space * self.reference_snapshot.values))

        input_ = torch.from_numpy(np.vstack([
            np.vstack([s.space, np.ones(shape=(s.space.shape[0],))*p.values]).T
            for p, s in db._pairs 
        ])).float()
        output_ = torch.from_numpy(np.vstack([
            self.reference_snapshot.space.reshape(-1, 1)
            for _ in db._pairs
        ]))

        self.shift_network._build_model(input_, output_)
        optimizer = self.shift_network.optimizer(
            self.shift_network.model.parameters(),
            lr=self.shift_network.lr,
            weight_decay=self.shift_network.l2_regularization)

        n_epoch = 1
        flag = True
        while flag:

            shifts = self.shift_network.model(input_).float()
            loss = torch.tensor([0.0])
            for (_, snap), shift in zip(db._pairs, np.split(shifts, len(db))):

                tensor_space = torch.from_numpy(snap.space)
                tensor_values = torch.from_numpy(snap.values)

                translated_space = tensor_space - shift.reshape(snap.space.shape)
                translated_space = translated_space.float()
                interpolated_reference_values = self.interp_network.model(translated_space.reshape(-1, 1)).float().flatten()

                diff = torch.mean(
                    (tensor_values - interpolated_reference_values)**2)

                if self.barycenter_loss:
                    snap_center = torch.mean(translated_space * tensor_values)
                    diff += self.barycenter_loss*(ref_center - snap_center)**2

                loss += diff

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scalar_loss = loss.item()
            self.shift_network.loss_trend.append(scalar_loss)

            for criteria in self.shift_network.stop_training:
                if isinstance(criteria, int):  # stop criteria is an integer
                    if n_epoch == criteria:
                        flag = False
                elif isinstance(criteria, float):  # stop criteria is float
                    if scalar_loss < criteria:
                        flag = False

            if (flag is False or
                    n_epoch == 1 or n_epoch % self.shift_network.frequency_print == 0):
                print(f'[epoch {n_epoch:6d}]\t{scalar_loss:e}')

            n_epoch += 1

    def fom_preprocessing(self, rom):
        db = rom._full_database

        reference_snapshot = db._pairs[self.reference_index][1]
        self.reference_snapshot = reference_snapshot

        self._train_interp_network()
        self._train_shift_network(db)

        for param, snap in db._pairs:
            input_shift = np.hstack([
                snap.space.reshape(-1, 1),
                np.ones(shape=(snap.space.shape[0], 1))*param.values])
            shift = self.shift_network.predict(input_shift)
            
            self.interpolator.fit(
                snap.space.reshape(-1, 1) - shift,
                snap.values.reshape(-1, 1))

            snap.values = self.interpolator.predict(
                reference_snapshot.space.reshape(-1, 1)).flatten()

    def fom_postprocessing(self, rom):
        
        ref_space = self.reference_snapshot.space

        for param, snap in rom._full_database._pairs:
            input_shift = np.hstack([
                ref_space.reshape(-1, 1),
                np.ones(shape=(ref_space.shape[0], 1))*param.values])
            shift = self.shift_network.predict(input_shift)
            snap.space = ref_space + shift.flatten()
            snap.space = snap.space.flatten()
