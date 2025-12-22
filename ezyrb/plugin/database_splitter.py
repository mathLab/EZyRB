
from .plugin import Plugin
from ..database import Database


class DatabaseSplitter(Plugin):
    """
    Plugin for splitting the database into training, test, validation, and prediction sets.
    
    This plugin automatically splits the database according to specified ratios
    before the fitting process begins.
    
    :param float train: Ratio or number of samples for training set. Default is 0.9.
    :param float test: Ratio or number of samples for test set. Default is 0.1.
    :param float validation: Ratio or number of samples for validation set. Default is 0.0.
    :param float predict: Ratio or number of samples for prediction set. Default is 0.0.
    :param int seed: Random seed for reproducibility. Default is None.
    
    :Example:
    
        >>> from ezyrb import ReducedOrderModel as ROM
        >>> from ezyrb import POD, RBF, Database
        >>> from ezyrb.plugin import DatabaseSplitter
        >>> import numpy as np
        >>> params = np.random.rand(100, 2)
        >>> snapshots = np.random.rand(100, 50)
        >>> db = Database(params, snapshots)
        >>> pod = POD(rank=5)
        >>> rbf = RBF()
        >>> splitter = DatabaseSplitter(train=0.7, test=0.2, validation=0.1)
        >>> rom = ROM(db, pod, rbf, plugins=[splitter])
        >>> rom.fit()
    """


    def __init__(self, train=0.9, test=0.1, validation=0.0, predict=0.0,
            seed=None):
        """
        Initialize the DatabaseSplitter plugin.
        
        :param float train: Ratio for training set. Default is 0.9.
        :param float test: Ratio for test set. Default is 0.1.
        :param float validation: Ratio for validation set. Default is 0.0.
        :param float predict: Ratio for prediction set. Default is 0.0.
        :param int seed: Random seed. Default is None.
        """
        super().__init__()

        self.train = train
        self.test = test
        self.validation = validation
        self.predict = predict
        self.seed = seed

    def fit_preprocessing(self, rom):
        """
        Split the database before fitting begins.
        
        :param ReducedOrderModel rom: The ROM instance.
        """
        db = rom._database
        if isinstance(db, Database):
            train, test, validation, predict = db.split(
                [self.train, self.test, self.validation, self.predict],
                seed=self.seed
            )

        elif isinstance(db, dict):
            train, test, validation, predict = list(db.values())[0].split(
                [self.train, self.test, self.validation, self.predict],
                seed=self.seed
            )
            # TODO improve this splitting if needed (now only reading the database of
            # the first ROM)


        rom.train_full_database = train
        rom.test_full_database = test
        rom.validation_full_database = validation
        rom.predict_full_database = predict
        #print('train', train.snapshots_matrix.shape)
        #print('test', test.snapshots_matrix.shape)
        #print('validation', validation.snapshots_matrix.shape)
        #print('predict', predict.snapshots_matrix.shape)

class DatabaseDictionarySplitter(Plugin):
    """
    This plugin class is used to define the train, test, validation and predict
    databases when the databases are already split: train, test, validation and
    predict are already database objects stored in a dictionary. Given the desired keys
    of the dictionary as input, the plugin will assign the corresponding database
    objects to the train, test, validation and predict attributes of the ROM.
    
    :Example:
    
        >>> from ezyrb import ReducedOrderModel as ROM
        >>> from ezyrb import POD, RBF, Database
        >>> from ezyrb.plugin import DatabaseDictionarySplitter
        >>> db_dict = {
        ...     'train': Database(train_params, train_snaps),
        ...     'test': Database(test_params, test_snaps)
        ... }
        >>> pod = POD(rank=5)
        >>> rbf = RBF()
        >>> splitter = DatabaseDictionarySplitter(train_key='train', test_key='test')
        >>> rom = ROM(db_dict['train'], pod, rbf, plugins=[splitter])
        >>> rom.fit()
    """
    
    def __init__(self, train_key=None, test_key=None, validation_key=None,
                 predict_key=None):
        """
        Initialize the DatabaseDictionarySplitter plugin.
        
        :param str train_key: Dictionary key for training database. Default is None.
        :param str test_key: Dictionary key for test database. Default is None.
        :param str validation_key: Dictionary key for validation database. Default is None.
        :param str predict_key: Dictionary key for prediction database. Default is None.
        """
        super().__init__()
        self.train_key = train_key
        self.test_key = test_key
        self.validation_key = validation_key
        self.predict_key = predict_key

    def fit_preprocessing(self, rom):
        """
        Assign the database splits from the dictionary before fitting.
        
        :param ReducedOrderModel rom: The ROM instance.
        :raises ValueError: If the database is not a dictionary.
        """
        db = rom._database
        if isinstance(db, dict):
            if self.train_key is not None:
                rom.train_full_database = db[self.train_key]
            if self.test_key is not None:
                rom.test_full_database = db[self.test_key]
            if self.validation_key is not None:
                rom.validation_full_database = db[self.validation_key]
            if self.predict_key is not None:
                rom.predict_full_database = db[self.predict_key]
        else:
            raise ValueError("The database must be a dictionary of databases.")
           