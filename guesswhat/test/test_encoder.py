import unittest
import torch
import torch.nn as nn
from models import Encoder


class TestEncoder(unittest.TestCase):

    def test_cells(self):

        for cell in [nn.RNN, nn.GRU, nn.LSTM]:
            Encoder(10, 5, cell=cell)

        with self.assertRaises(ValueError):
            Encoder(10, 5, object)

    def test_forward(self):

        input = torch.Tensor(4, 8, 10).random_()
        lengths = torch.Tensor([3, 8, 2, 7]).long()

        # GRU
        encoder = Encoder(10, 5, cell=nn.GRU)
        outputs, last_hidden = encoder(input, lengths)
        self.assertEqual(list(outputs.size()), [4, 8, 5])
        self.assertEqual(list(last_hidden.size()), [1, 4, 5])

        # LSTM
        encoder = Encoder(10, 5, cell=nn.LSTM)
        outputs, last_hidden = encoder(input, lengths)
        self.assertEqual(list(outputs.size()), [4, 8, 5])
        self.assertIsInstance(last_hidden, tuple)
        self.assertEqual(list(last_hidden[0].size()), [1, 4, 5])
        self.assertEqual(list(last_hidden[1].size()), [1, 4, 5])

    def test_different_batch_dimensions(self):

        input = torch.Tensor(4, 8, 10).random_()
        lengths = torch.Tensor([3, 8, 2]).long()
        encoder = Encoder(10, 5, cell=nn.GRU)

        with self.assertRaises(AssertionError):
            outputs, last_hidden = encoder(input, lengths)
