
from .plugin import Plugin


class DatabaseSplitter(Plugin):


    def __init__(self, train=0.9, test=0.1, validation=0.0, predict=0.0,
            seed=None):
        super().__init__()

        if sum([train, test, validation, predict]) != 1.0:
            raise ValueError('The sum of the ratios must be equal to 1.0')

        self.train = train
        self.test = test
        self.validation = validation
        self.predict = predict
        self.seed = seed

    def fit_preprocessing(self, rom):
        db = rom._database
        train, test, validation, predict = db.split(
            [self.train, self.test, self.validation, self.predict],
            seed=self.seed
        )

        rom.train_full_database = train
        rom.test_full_database = test
        rom.validation_full_database = validation
        rom.predict_full_database = predict
        print('train', train.snapshots_matrix.shape)
        print('test', test.snapshots_matrix.shape)
        print('validation', validation.snapshots_matrix.shape)
        print('predict', predict.snapshots_matrix.shape)