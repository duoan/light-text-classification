# _*_ coding: utf-8 _*_
from argparse import ArgumentParser

import torch
from torchtext import data, datasets

from vocab import LocalVectors

from models import *

from torch.optim import SGD
from torch.utils.data import DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from tqdm import tqdm


def get_data_loaders(batch_size=32):
  tokenize = lambda x: x.split()
  TEXT = data.Field(
      sequential=True,
      tokenize=tokenize,
      lower=True,
      include_lengths=True,
      batch_first=True,
      fix_length=200)
  LABEL = data.LabelField(dtype=torch.float)
  print('Load IMDB dataset')
  train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
  print('TEXT build vocab')
  TEXT.build_vocab(
      train_data,
      vectors=LocalVectors(
          '/Users/duoan/nbs/quora-insincere-questions-classification/input/embeddings/glove.840B.300d/glove.840B.300d.txt'
      ))
  print('LABEL build vocab')
  LABEL.build_vocab(train_data)

  word_embeddings = TEXT.vocab.vectors
  print('Length of TEXT Vocabulary: {}'.format(len(TEXT.vocab)))
  print('Vector size of TEXT Vocabulary: {}'.format(TEXT.vocab.vectors.size()))
  print('LABEL Length: {}'.format(len(LABEL.vocab)))

  train_data, valid_data = train_data.split()
  train_iter, valid_iter, test_iter = data.BucketIterator.splits(
      (train_data, valid_data, test_data),
      batch_size=batch_size,
      sort_key=lambda x: len(x.text),
      repeat=False,
      shuffle=True)
  vocab_size = len(TEXT.vocab)
  print('finished get data loaders')
  return vocab_size, word_embeddings, train_iter, valid_iter, test_iter


def run(batch_size, epochs, lr, momentum, log_interval):
  vocab_size, word_embeddings, train_iter, valid_iter, test_iter = get_data_loaders(
      batch_size)
  model = LSTMClassifier(32, 2, 256, vocab_size, 300, word_embeddings)
  device = 'cpu'

  if torch.cuda.is_available():
    device = 'cuda'

  optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
  trainer = create_supervised_trainer(
      model, optimizer, F.nll_loss, device=device)
  evaluator = create_supervised_evaluator(
      model,
      metrics={
          'accuracy': Accuracy(),
          'nll': Loss(F.nll_loss)
      },
      device=device)

  desc = "ITERATION - loss: {:.2f}"
  pbar = tqdm(
      initial=0, leave=False, total=len(train_iter), desc=desc.format(0))

  @trainer.on(Events.ITERATION_COMPLETED)
  def log_training_loss(engine):
    iter = (engine.state.iteration - 1) % len(train_iter) + 1
    if iter % log_interval == 0:
      pbar.desc = desc.format(engine.state.output)
      pbar.update(log_interval)

  @trainer.on(Events.EPOCH_COMPLETED)
  def log_training_results(engine):
    pbar.refresh()
    evaluator.run(train_iter)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_nll = metrics['nll']
    tqdm.write(
        "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        .format(engine.state.epoch, avg_accuracy, avg_nll))

  @trainer.on(Events.EPOCH_COMPLETED)
  def log_validation_results(engine):
    evaluator.run(valid_iter)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_nll = metrics['nll']
    tqdm.write(
        "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        .format(engine.state.epoch, avg_accuracy, avg_nll))

    pbar.n = pbar.last_print_n = 0

  trainer.run(train_iter, max_epochs=epochs)
  pbar.close()


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=64,
      help='input batch size for training (default: 64)')
  parser.add_argument(
      '--val_batch_size',
      type=int,
      default=1000,
      help='input batch size for validation (default: 1000)')
  parser.add_argument(
      '--epochs',
      type=int,
      default=10,
      help='number of epochs to train (default: 10)')
  parser.add_argument(
      '--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
  parser.add_argument(
      '--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
  parser.add_argument(
      '--log_interval',
      type=int,
      default=10,
      help='how many batches to wait before logging training status')

  args = parser.parse_args()

  run(args.batch_size, args.epochs, args.lr, args.momentum, args.log_interval)
