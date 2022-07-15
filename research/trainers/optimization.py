"""Training and test loops
"""
from time import time
from math import floor

import torch
from torch.utils.data import DataLoader
from aihwkit.nn import AnalogSequential

class Tester:
    """Implementation of test loop.
    """
    def __init__(self, test_data, batch_size=64, loss_fn=None, batch_first=False,
            test_rpu_config=None, max_accuracy=False) -> None:
        """Test neural network model.
        Args:
            model (torch.nn.Module): neural network model.
            test_data (Dataset): test dataset.
            batch_size (int, optional): batch size. Defaults to 64.
            loss_fn_string (str, optional): loss function string. Defaults to "CrossEntropyLoss".
            batch_first (bool, optinal): whether batch_size is first dimension or not. Defaults to True.
                                         Must be False for AnalogRNN.
        """
        # wrap an iterable around the Dataset to enable easy access to the samples
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.batch_first = batch_first

        self.test_rpu_config = test_rpu_config

        self.max_accuracy = 0. if max_accuracy else None
        self.max_state_dict = None

        self.reset_stats()
    

    def __call__(self, model) -> None:
        self.test_loop(model)


    def test_loop(self, model) -> None:
        """Iterate over the test dataset to check if model performance is improving.
        """
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        test_loss, correct = 0, 0

        model.eval()
        # load test_rpu_config, program and drift weights
        if isinstance(model, AnalogSequential):
            model.load_rpu_config(self.test_rpu_config)
            model.program_analog_weights()
            model.drift_analog_weights()

        # disable gradient tracking (forward pass)
        with torch.no_grad():
            for X, y in self.test_dataloader: # batch_first in DataLoader

                if not self.batch_first:
                    X = torch.transpose(X, 0, 1) # swap batch_size and sequence_length

                pred = model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        self.average_losses.append(test_loss)
        self.accuracies.append(correct)
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

        if self.max_accuracy is not None and self.max_accuracy < correct:
            self.max_accuracy = correct
            self.max_state_dict = model.state_dict()

        if isinstance(model, AnalogSequential):
            model.unprogram_analog_weights()
        
    
    def reset_stats(self):
        self.accuracies = []
        self.average_losses = []


class Trainer(Tester):
    """Full implementation of optimization loop.
    """
    def __init__(self, training_data, test_data, learning_rate=1e-3, batch_size=64, epochs=10,
        loss_fn=None, optimizer=None, batch_first=False,
        training_rpu_config=None, test_rpu_config=None,
        max_accuracy=False) -> None:
        """Constructor.
        Args:
            model (torch.nn.Module): neural network model.
            training_data (Dataset): training dataset.
            test_data (Dataset): test dataset.
            learning_rate (float, optional): learning rate. Defaults to 1e-3.
            batch_size (int, optional): batch size. Defaults to 64.
            epochs (int, optional): number of epochs. Defaults to 10.
            batch_first (bool, optinal): whether batch_size is first dimension or not. Defaults to True.
                                         Must be False for AnalogRNN.
        """
        super().__init__(test_data, batch_size, loss_fn, batch_first, test_rpu_config, max_accuracy)
        # wrap an iterable around the Dataset to enable easy access to the samples
        self.training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer

        self.training_rpu_config = training_rpu_config
    

    def __call__(self, model, n_step=1) -> None:
        """Optimization loop.
        Returns:
            torch.nn.Module: trained neural network model.
        """
        time_init = time()
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.training_loop(model, n_step)
            time_now = time() - time_init
            print(f"Time: {time_now:>4f} s\n")
        print("Done!")
    

    def training_state_dict(self) -> dict:
        state_dict = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "loss_fn": self.loss_fn.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
            "accuracies": self.accuracies,
            "average_losses": self.average_losses
        }
        return state_dict


    def training_loop(self, model, n_step=1) -> None:
        """Iterate over the training dataset and try to converge to optimal parameters.
        """
        size = len(self.training_dataloader.dataset)
        num_batches = len(self.training_dataloader)

        step = floor(num_batches / n_step)

        model.train()

        if isinstance(model, AnalogSequential):
            model.load_rpu_config(self.training_rpu_config)

        for batch, (X, y) in enumerate(self.training_dataloader): # batch_first in DataLoader
            batch += 1
            # Compute prediction and loss
            batch_size = len(X)
            if not self.batch_first:
                X = torch.transpose(X, 0, 1) # swap batch_size and sequence_length

            pred = model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad() # reset the gradients of model parameters
            loss.backward() # backpropagate the prediction loss
            self.optimizer.step() # adjust the parameters

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            if batch % step == 0:
                self.test_loop(model)
