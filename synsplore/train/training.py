from synsplore.model.main import SynModule
from typing import Dict, Optional
import torch
from tqdm import tqdm
import logging
import re


class Training:
    """Base class for training models."""
    
    def __init__(self, model, data, optimizer, scheduler=None,
                 logger_dir=None):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = next(model.parameters()).device if next(model.parameters(), None) is not None else 'cpu'
        self.model.to(self.device)

    def train(self, epochs: int):
        """Train the model for a specified number of epochs."""
        for epoch in range(epochs):
            total_loss = 0.0
            with tqdm(self.data, unit="batch") as bar:
                for b_idx, batch in enumerate(bar):
                    batch = batch["syndata"]
                    bar.set_description(f"Epoch {epoch}")

                    (cls_pred, r_pred, rxn_pred, 
                     loss_cls, loss_r, loss_rxn, 
                     batch_loss, metrics_outputs) = self.train_step(batch)
                    
                    # print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
                    cls_metrics_logs = ""
                    for n, m in metrics_outputs["cls_metrics"].items():
                        if n == "cm":
                            cls_metrics_logs += f"cls_{n}: {m.flatten().tolist()}, " 
                        else:
                            cls_metrics_logs += f"cls_{n}: {m.item():.4f}, " 

                    r_metrics_logs = ""
                    for n, m in metrics_outputs["r_metrics"].items():
                        if n == "cm":
                            r_metrics_logs += f"r_{n}: {m.flatten().tolist()}, " 
                        else:
                            r_metrics_logs += f"r_{n}: {m.item():.4f}, " 

                    rxn_metrics_logs = ""
                    for n, m in metrics_outputs["rxn_metrics"].items():
                        if n == "cm":
                            rxn_metrics_logs += f"rxn_{n}: {m.flatten().tolist()}, " 
                        else:
                            rxn_metrics_logs += f"rxn_{n}: {m.item():.4f}, " 

                    total_loss += batch_loss
                    logging.info(re.sub(' +', ' ', 
                                        f"Epoch: {epoch + 1}/{epochs}, \
                                        Batch: {b_idx + 1}, \
                                        Loss: {batch_loss:.4f}, \
                                        cls_loss: {loss_cls:.4f}, \
                                        r_loss: {loss_r:.4f}, \
                                        rxn_loss: {loss_rxn:.4f}, \
                                        {''.join(cls_metrics_logs)}, \
                                        {''.join(r_metrics_logs)}, \
                                        {''.join(rxn_metrics_logs)}, \
                                        "))
                    bar.set_postfix(
                        r_loss = float(loss_r),
                        cls_loss = float(loss_cls),
                        rxn_loss = float(loss_rxn),
                        loss = float(batch_loss),
                    )
        
        logging.info("Training complete.")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        cls_pred, r_pred, rxn_pred = self.model(batch)
        loss_cls, loss_r, loss_rxn = self.model.get_loss(batch)
        metrics_outputs = self.model.get_metrics(batch)

        ## log the loss functions
        total_loss = loss_cls/loss_cls.detach() + loss_r/loss_r.detach() + loss_rxn/loss_rxn.detach()
        
        total_loss.backward()
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        return (cls_pred, r_pred, rxn_pred, 
                loss_cls, loss_r, loss_rxn, 
                total_loss.item(), metrics_outputs)