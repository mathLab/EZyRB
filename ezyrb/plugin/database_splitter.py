
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

