
from .plugin import Plugin
from ..database import Database


class DatabaseSplitter(Plugin):


    def __init__(self, train=0.9, test=0.1, validation=0.0, predict=0.0,
            seed=None):
        super().__init__()

        self.train = train
        self.test = test
        self.validation = validation
        self.predict = predict
        self.seed = seed

    def fit_preprocessing(self, rom):
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
    """
    
    def __init__(self, train_key=None, test_key=None, validation_key=None,
                 predict_key=None):
        super().__init__()
        self.train_key = train_key
        self.test_key = test_key
        self.validation_key = validation_key
        self.predict_key = predict_key

    def fit_preprocessing(self, rom):
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
           