import numpy as np
from unittest import TestCase
from ezyrb import Database
from ezyrb.plugin.database_splitter import DatabaseSplitter, DatabaseDictionarySplitter

class DummyROM:
    train_full_database = None
    test_full_database = None
    validation_full_database = None
    predict_full_database = None

    def __init__(self, db):
        self._database = db


class TestDatabaseSplitter(TestCase):

    def test_split_integers_train_size(self):
        db = Database(np.random.uniform(size=(100, 2)),
                      np.random.uniform(size=(100, 5)))
        rom = DummyROM(db)
        splitter = DatabaseSplitter(train=80, test=20, validation=0, predict=0)
        splitter.fit_preprocessing(rom)
        self.assertEqual(len(rom.train_full_database), 80)

    def test_split_integers_test_size(self):
        db = Database(np.random.uniform(size=(100, 2)),
                      np.random.uniform(size=(100, 5)))
        rom = DummyROM(db)
        splitter = DatabaseSplitter(train=80, test=20, validation=0, predict=0)
        splitter.fit_preprocessing(rom)
        self.assertEqual(len(rom.test_full_database), 20)

    def test_split_integers_validation_predict_empty(self):
        db = Database(np.random.uniform(size=(100, 2)),
                      np.random.uniform(size=(100, 5)))
        rom = DummyROM(db)
        splitter = DatabaseSplitter(train=80, test=20, validation=0, predict=0)
        splitter.fit_preprocessing(rom)
        self.assertEqual(len(rom.validation_full_database), 0)
        self.assertEqual(len(rom.predict_full_database), 0)

    def test_split_integers_total_conserved(self):
        db = Database(np.random.uniform(size=(100, 2)),
                      np.random.uniform(size=(100, 5)))
        rom = DummyROM(db)
        splitter = DatabaseSplitter(train=70, test=20, validation=5, predict=5)
        splitter.fit_preprocessing(rom)
        total = (len(rom.train_full_database) +
                 len(rom.test_full_database) +
                 len(rom.validation_full_database) +
                 len(rom.predict_full_database))
        self.assertEqual(total, 100)

    def test_split_integers_returns_database_instances(self):
        db = Database(np.random.uniform(size=(100, 2)),
                      np.random.uniform(size=(100, 5)))
        rom = DummyROM(db)
        splitter = DatabaseSplitter(train=80, test=20, validation=0, predict=0)
        splitter.fit_preprocessing(rom)
        self.assertIsInstance(rom.train_full_database, Database)
        self.assertIsInstance(rom.test_full_database, Database)
        self.assertIsInstance(rom.validation_full_database, Database)
        self.assertIsInstance(rom.predict_full_database, Database)

    def test_split_integers_inconsistent_chunks_raises(self):
        db = Database(np.random.uniform(size=(100, 2)),
                      np.random.uniform(size=(100, 5)))
        rom = DummyROM(db)
        splitter = DatabaseSplitter(train=70, test=20, validation=0, predict=0)
        with self.assertRaises(ValueError):
            splitter.fit_preprocessing(rom)


    def test_split_floats_total_conserved(self):
        db = Database(np.random.uniform(size=(100, 2)),
                      np.random.uniform(size=(100, 5)))
        rom = DummyROM(db)
        splitter = DatabaseSplitter(train=0.7, test=0.2, validation=0.05,
                                    predict=0.05, seed=0)
        splitter.fit_preprocessing(rom)
        total = (len(rom.train_full_database) +
                 len(rom.test_full_database) +
                 len(rom.validation_full_database) +
                 len(rom.predict_full_database))
        self.assertEqual(total, 100)

    def test_split_floats_returns_database_instances(self):
        db = Database(np.random.uniform(size=(100, 2)),
                      np.random.uniform(size=(100, 5)))
        rom = DummyROM(db)
        splitter = DatabaseSplitter(train=0.8, test=0.2, seed=0)
        splitter.fit_preprocessing(rom)
        self.assertIsInstance(rom.train_full_database, Database)
        self.assertIsInstance(rom.test_full_database, Database)

    def test_split_floats_inconsistent_ratios_raises(self):
        db = Database(np.random.uniform(size=(100, 2)),
                      np.random.uniform(size=(100, 5)))
        rom = DummyROM(db)
        splitter = DatabaseSplitter(train=0.7, test=0.2, validation=0.0,
                                    predict=0.0)
        with self.assertRaises(ValueError):
            splitter.fit_preprocessing(rom)

   
    def test_split_seed_reproducibility(self):
        db = Database(np.random.uniform(size=(100, 2)),
                      np.random.uniform(size=(100, 5)))
        rom1 = DummyROM(db)
        DatabaseSplitter(train=0.8, test=0.2, seed=42).fit_preprocessing(rom1)

        rom2 = DummyROM(db)
        DatabaseSplitter(train=0.8, test=0.2, seed=42).fit_preprocessing(rom2)

        np.testing.assert_array_equal(
            rom1.train_full_database.parameters_matrix,
            rom2.train_full_database.parameters_matrix,
        )

    def test_split_different_seeds_differ(self):
        db = Database(np.random.uniform(size=(100, 2)),
                      np.random.uniform(size=(100, 5)))
        rom1 = DummyROM(db)
        DatabaseSplitter(train=0.8, test=0.2, seed=0).fit_preprocessing(rom1)

        rom2 = DummyROM(db)
        DatabaseSplitter(train=0.8, test=0.2, seed=99).fit_preprocessing(rom2)

        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(
                rom1.train_full_database.parameters_matrix,
                rom2.train_full_database.parameters_matrix,
            )

    def test_split_dict_database_explicit_flattening(self):
        db_a = Database(np.random.uniform(size=(100, 2)),
                        np.random.uniform(size=(100, 5)))
        db_b = Database(np.random.uniform(size=(50, 2)),
                        np.random.uniform(size=(50, 5)))
        
        rom = DummyROM({'a': db_a, 'b': db_b})
        
        splitter = DatabaseSplitter(train=80, test=20, validation=0, predict=0)
        splitter.fit_preprocessing(rom)
        
        self.assertIsInstance(rom.train_full_database, Database)
        
        self.assertEqual(len(rom.train_full_database), 80)


class TestDatabaseDictionarySplitter(TestCase):

    def _make_dict_rom(self):
        db_train = Database(np.random.uniform(size=(60, 2)),
                            np.random.uniform(size=(60, 5)))
        db_test = Database(np.random.uniform(size=(20, 2)),
                           np.random.uniform(size=(20, 5)))
        db_val = Database(np.random.uniform(size=(10, 2)),
                          np.random.uniform(size=(10, 5)))
        db_pred = Database(np.random.uniform(size=(10, 2)),
                           np.random.uniform(size=(10, 5)))
        db_dict = {
            'train': db_train,
            'test': db_test,
            'val': db_val,
            'pred': db_pred,
        }
        return DummyROM(db_dict), db_dict

    def test_train_key_assigned(self):
        rom, db_dict = self._make_dict_rom()
        DatabaseDictionarySplitter(train_key='train').fit_preprocessing(rom)
        self.assertEqual(len(rom.train_full_database), 60)

    def test_test_key_assigned(self):
        rom, db_dict = self._make_dict_rom()
        DatabaseDictionarySplitter(test_key='test').fit_preprocessing(rom)
        self.assertEqual(len(rom.test_full_database), 20)

    def test_validation_key_assigned(self):
        rom, db_dict = self._make_dict_rom()
        DatabaseDictionarySplitter(validation_key='val').fit_preprocessing(rom)
        self.assertEqual(len(rom.validation_full_database), 10)

    def test_predict_key_assigned(self):
        rom, db_dict = self._make_dict_rom()
        DatabaseDictionarySplitter(predict_key='pred').fit_preprocessing(rom)
        self.assertEqual(len(rom.predict_full_database), 10)

    def test_all_keys_assigned(self):
        rom, db_dict = self._make_dict_rom()
        splitter = DatabaseDictionarySplitter(
            train_key='train', test_key='test',
            validation_key='val', predict_key='pred',
        )
        splitter.fit_preprocessing(rom)
        self.assertEqual(len(rom.train_full_database), 60)
        self.assertEqual(len(rom.test_full_database), 20)
        self.assertEqual(len(rom.validation_full_database), 10)
        self.assertEqual(len(rom.predict_full_database), 10)

    def test_assigned_database_is_same_object(self):
        rom, db_dict = self._make_dict_rom()
        splitter = DatabaseDictionarySplitter(
            train_key='train', test_key='test',
        )
        splitter.fit_preprocessing(rom)
        self.assertIs(rom.train_full_database, db_dict['train'])
        self.assertIs(rom.test_full_database, db_dict['test'])

    def test_unset_key_leaves_attribute_none(self):
        rom, _ = self._make_dict_rom()
        DatabaseDictionarySplitter(train_key='train').fit_preprocessing(rom)
        self.assertIsNone(rom.test_full_database)
        self.assertIsNone(rom.validation_full_database)
        self.assertIsNone(rom.predict_full_database)

    def test_non_dict_database_raises(self):
        db = Database(np.random.uniform(size=(10, 2)),
                      np.random.uniform(size=(10, 5)))
        rom = DummyROM(db)
        splitter = DatabaseDictionarySplitter(train_key='train')
        with self.assertRaises(ValueError):
            splitter.fit_preprocessing(rom)