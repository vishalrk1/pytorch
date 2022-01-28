import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm

class Trainer(object):
    def __init__(self, model, device, loss_fn=None, optimizer=None, scheduler=None, save_path=None):

        # Set params
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = save_path

    def train_step(self, dataloader):
        """Train step."""
        # Set model to train mode
        self.model.train()
        loss = 0.0
        acc = 0.0

        # Iterate over train batches
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

            # Step
            batch = [item.to(self.device) for item in batch]  # Set device
            inputs, targets = batch[:-1], batch[-1]
            self.optimizer.zero_grad()  # Reset gradients
            z = self.model(inputs)  # Forward pass
            J = self.loss_fn(z, targets)  # Define loss
            J.backward()  # Backward pass
            self.optimizer.step()  # Update weights

            # Cumulative Metrics
            loss += (J.detach().item() - loss) / (i + 1)
            acc += self.calculate_accuracy(z, targets)

        return loss, acc

    def eval_step(self, dataloader):
        """Validation or test step."""
        # Set model to eval mode
        self.model.eval()
        loss = 0.0
        acc = 0.0
        y_trues, y_probs = [], []

        # Iterate over val batches
        with torch.inference_mode():
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

                # Step
                batch = [item.to(self.device) for item in batch]  # Set device
                inputs, y_true = batch[:-1], batch[-1]
                z = self.model(inputs)  # Forward pass
                J = self.loss_fn(z, y_true).item()

                # Cumulative Metrics
                loss += (J - loss) / (i + 1)
                acc += self.calculate_accuracy(z, y_true)

                # Store outputs
                y_prob = F.softmax(z).cpu().numpy()
                y_probs.extend(y_prob)
                y_trues.extend(y_true.cpu().numpy())

        return loss, acc, np.vstack(y_trues), np.vstack(y_probs)

    def predict_step(self, dataloader):
        """Prediction step."""
        # Set model to eval mode
        self.model.eval()
        y_probs = []

        # Iterate over val batches
        with torch.inference_mode():
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

                # Forward pass w/ inputs
                inputs, targets = batch[:-1], batch[-1]
                z = self.model(inputs)

                # Store outputs
                y_prob = F.softmax(z).cpu().numpy()
                y_probs.extend(y_prob)

        return np.vstack(y_probs)

    def calculate_accuracy(self, output, labels):
      output_class = torch.softmax(output, dim=1).argmax(dim=1)
      train_acc = (output_class == labels).sum().item()/len(output)
      return train_acc

    def train(self, num_epochs, patience, train_dataloader, val_dataloader):
        best_val_loss = np.inf
        for epoch in range(num_epochs):
            print(f"<----- Epoch: {epoch+1} ----->")
            # Steps
            train_loss, train_acc = self.train_step(dataloader=train_dataloader)
            val_loss, val_acc, _, _ = self.eval_step(dataloader=val_dataloader)
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                if self.save_path is not None:
                  print('Saving Model!!')
                  torch.save(self.model.state_dict(), self.save_path)
                _patience = patience  # reset _patience
            else:
                _patience -= 1
            if not _patience:  # 0
                print("Stopping early!")
                break

            train_acc = train_acc / len(train_dataloader)
            val_acc = val_acc / len(val_dataloader)

            # Logging
            print(
                f"train Loss: {train_loss:.5f},\t"
                f"train Accuracy: {train_acc:.2f},\n"
                f"val Loss: {val_loss:.5f}, \t"
                f"val Accuracy: {val_acc:.2f}, \n"
                f"lr: {self.optimizer.param_groups[0]['lr']:.2E}, \t"
                f"_patience: {_patience}"
                "\n"
            )
        return best_model