import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from tqdm import tqdm
from models.losses.detr_loss import compute_sample_loss
from torch.optim.lr_scheduler import MultiStepLR
#from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.scheduler.cosine_lr import CosineLRScheduler
#scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
class DETRTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epochs: int,
        batch_size: int,
        log_freq: int = 1,
        save_freq: int = 10,
        weight_decay: float = 1e-4, #0.05 for large dataset
        checkpoint_dir: str = "ckpts",
        freeze_backbone: bool = False,
        backbone_lr: float = 1e-5,
        transformer_lr: float = 1e-4,
        num_queries: int = 100,
        empty_class_id: int = 0,
    ):
        """
        Initializes the DETR trainer class.

        Public API:
        - train() : Start the training
        - visualize_losses() : Plot the training losses and save plots

        Args:
            model: The DETR model to train
            train_loader: The Data Loader for the training data set
            val_loader: The Data Loader for the validation data set
            device: The device to run the model on
            epochs: The number of epochs to train for
            batch_size: The number of samples in a batch
            log_freq: How often to log the loss (default: 1)
            save_freq: How often to save the model (default: 10)
            weight_decay: The weight decay for the AdamW optimizer (default: 1e-4)
            checkpoint_dir: The directory to save the model checkpoints (default: "ckpts")
            freeze_backbone: Whether to freeze the backbone during training (default: False)
            backbone_lr: The learning rate for the backbone (default: 1e-5)
            transformer_lr: The learning rate for the transformer (default: 1e-4)
            num_queries: The number of object queries (default: 100)
            empty_class_id: The class id for the empty class (default: 0)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.num_train_batches = len(self.train_loader)
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.num_queries = num_queries
        self.empty_class_id = empty_class_id
        
        # History objects to hold training time metrics
        self.hist = []
        self.hist_detailed_losses = []

        # Create the optimizer with different learning rates for backbone/Transformer head and
        # optionally freeze the backbone during training.
        backbone_params = [p for n, p in model.named_parameters() if "backbone." in n]

        if freeze_backbone:
            print("Freezing CNN backbone...")
            for p in model.backbone.parameters():
                p.requires_grad = False
        else:
            # This is needed to re-enable the training of the backbone in case a previous
            # training iteration kept it frozen...
            for p in model.backbone.parameters():
                p.requires_grad = True
        print(f"Backbone is trainable: {not freeze_backbone}")

        transformer_params = [
            p for n, p in model.named_parameters() if "backbone." not in n
        ]

        self.optimizer = AdamW(
            [
                {"params": transformer_params, "lr": transformer_lr},
                {"params": backbone_params, "lr": backbone_lr},
            ],
            weight_decay=weight_decay,eps=1e-8,betas=(0.9, 0.999)
        )
        #self.scheduler = MultiStepLR(self.optimizer, milestones=[int(0.6 * epochs), int(0.8 * epochs)],gamma=0.1)
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.epochs * len(self.train_loader)//2,
            lr_min=1e-6,
            warmup_lr_init=1e-7,
            warmup_t=int(self.epochs * len(self.train_loader)*0.06),
            cycle_limit=2,
            t_in_epochs=False,
            warmup_prefix=True
        )
        
        # Log the number of total trainable parameters
        nparams = (
            sum([p.nelement() for p in model.parameters() if p.requires_grad]) / 1e6
        )
        print(f"DETR trainable parameters: {nparams:.1f}M")
        bparams = (
            sum([p.nelement() for p in model.parameters() if not p.requires_grad]) / 1e6
        )
        print(f"Backbone parameters count: {bparams:.1f}M")
        # Create the checkpoint dir if it does not exist...
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def compute_loss(self, o_bbox, t_bbox, o_cl, t_cl, t_mask):
        """
        Computes the total loss for a single sample (image and corresponding GT labels).

        Args:
            o_bbox (torch.Tensor): The predicted bounding boxes (Shape: torch.Size([100, 4]))
            t_bbox (torch.Tensor): The ground truth bounding boxes (Shape: torch.Size([100, 4]))
            o_cl (torch.Tensor): The predicted class labels (Shape: torch.Size([100, num_classes]))
            t_cl (torch.Tensor): The ground truth class labels (Shape: torch.Size([100]))
            t_mask (torch.Tensor): The mask for the ground truth bounding boxes (Shape: torch.Size([100]))

        Returns:
            torch.Tensor: The total loss for the sample
        """
        return compute_sample_loss(
            o_bbox,
            t_bbox,
            o_cl,
            t_cl,
            t_mask,
            n_queries=self.num_queries,
            empty_class_id=self.empty_class_id,
            device=self.device,
        )

    def log_epoch_losses(self, epoch, losses, class_losses, box_losses, giou_losses):
        """Logs and stores loss values for an epoch based on the set log frequency.

        Args:
            epoch(int) : Current epoch idx
            losses(torch.Tensor): The tensor holding the total DETR losses objects (per-batch)
            class_losses(torch.Tensor): The tensor holding the class losses objects (per-batch)
            box_losses(torch.Tensor): The tensor holding the bounding box L1 losses objects (per-batch)
            giou_losses(torch.Tensor): The tensor holding the GIoU objects (per-batch)
        """
        if (epoch + 1) % self.log_freq == 0:
            loss_avg = losses[-self.num_train_batches :].mean().item()
            epoch_loss_class = class_losses[-self.num_train_batches :].mean().item()
            epoch_loss_bbox = box_losses[-self.num_train_batches :].mean().item()
            epoch_loss_giou = giou_losses[-self.num_train_batches :].mean().item()

            print(f"Epoch: {epoch+1}/{self.epochs}, DETR Loss: {loss_avg:.4f}")
            print(
                f"→ Class Loss: {epoch_loss_class:.4f}, BBox Loss: {epoch_loss_bbox:.4f}, GIoU Loss: {epoch_loss_giou:.4f}"
            )

            self.hist.append(loss_avg)
            self.hist_detailed_losses.append(
                (epoch_loss_class, epoch_loss_bbox, epoch_loss_giou)
            )

    def save_checkpoint(self, epoch):
        """Saves model checkpoints and training history at specified intervals."""
        if (epoch + 1) % self.save_freq == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pth")
            state = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
            }
            torch.save(state, ckpt_path)
            
    def load_checkpoint(self, ckpt_path, load_optimizer=True):
        """Load model, optimizer, and histories."""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"🔁 Loaded checkpoint")
    
    def load_loss_history(self, hist_file=None, detail_hist_file=None):
        """
        Loads training loss and detailed loss history from .npy files and updates the corresponding attributes.

        Args:
            hist_file (str, optional): Path to the .npy file containing the total loss history.
            detail_hist_file (str, optional): Path to the .npy file containing detailed loss history
                                            (class loss, bbox loss, GIoU loss).
        """
        if hist_file:
            try:
                self.hist = np.load(hist_file).tolist()
                print(f"Loaded loss history from {hist_file}.")
            except Exception as e:
                print(f"Error loading loss history file: {e}")

        if detail_hist_file:
            try:
                self.hist_detailed_losses = np.load(detail_hist_file).tolist()
                print(f"Loaded detailed loss history from {detail_hist_file}.")
            except Exception as e:
                print(f"Error loading detailed loss history file: {e}")

    def visualize_losses(self, save_dir=None):
        """
        Plots training loss over epochs and optionally saves the figure.

        Args:
            save_dir (str, optional): Directory to save the plots. If None, it only displays the plots.
        """

        # Create save directory if it doesn't exist
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        epochs = np.arange(1, len(self.hist) + 1) * self.log_freq

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.hist, label="Total Loss", marker="o", linestyle="-")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid()

        if save_dir:
            plt.savefig(os.path.join(save_dir, "DETR_training_loss.png"))
        plt.show()

        # If detailed loss is provided, plot them separately
        if self.hist_detailed_losses:
            class_loss, bbox_loss, giou_loss = zip(*self.hist_detailed_losses)

            plt.figure(figsize=(10, 5))
            plt.plot(epochs, class_loss, label="Class Loss", linestyle="--")
            plt.plot(epochs, bbox_loss, label="BBox Loss", linestyle="--")
            plt.plot(epochs, giou_loss, label="GIoU Loss", linestyle="--")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Detailed Training Loss Over Epochs")
            plt.legend()
            plt.grid()

            if save_dir:
                plt.savefig(os.path.join(save_dir, "DETR_training_losses.png"))
            plt.show()

    def train(self):
        """Trains the DETR model for a specified number of epochs, with checkpoint/log callbacks."""
        torch.set_grad_enabled(True)
        self.model.train()
        print(
            f"Starting training for {self.epochs} epochs... Using device : {self.device}"
        )

        losses = torch.tensor([], device=self.device)
        class_losses = torch.tensor([], device=self.device)
        box_losses = torch.tensor([], device=self.device)
        giou_losses = torch.tensor([], device=self.device)

        # Clear the training history from previous trainings..
        self.hist = []
        self.hist_detailed_losses = []
        # Track minimal loss for checkpointing
        best_loss = float("inf")
        best_epoch = -1
        step=0
        for epoch in range(self.epochs):
            train_loader = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

            losses, class_losses, box_losses, giou_losses = [], [], [], []

            for input_, (tgt_cl, tgt_bbox, tgt_mask, _) in train_loader:
                input_ = input_.to(self.device, non_blocking=True)
                tgt_cl = tgt_cl.to(self.device, non_blocking=True)
                tgt_bbox = tgt_bbox.to(self.device, non_blocking=True)
                tgt_mask = tgt_mask.bool().to(self.device, non_blocking=True)

                class_preds, bbox_preds,_ = self.model(input_)

                loss = torch.tensor(0.0, device=self.device)
                loss_class_batch = torch.tensor(0.0, device=self.device)
                loss_bbox_batch = torch.tensor(0.0, device=self.device)
                loss_giou_batch = torch.tensor(0.0, device=self.device)

                num_dec_layers = class_preds.shape[1]

                for i in range(num_dec_layers):
                    o_bbox = bbox_preds[:, i].sigmoid()
                    o_cl = class_preds[:, i]

                    for o_bbox_i, t_bbox_i, o_cl_i, t_cl_i, t_mask_i in zip(
                        o_bbox, tgt_bbox, o_cl, tgt_cl, tgt_mask
                    ):
                        loss_class, loss_bbox, loss_giou = self.compute_loss(
                            o_bbox_i, t_bbox_i, o_cl_i, t_cl_i, t_mask_i
                        )
                        sample_loss = 1 * loss_class + 5 * loss_bbox + 2 * loss_giou
                        loss += sample_loss / self.batch_size / num_dec_layers
                        loss_class_batch += loss_class / self.batch_size / num_dec_layers
                        loss_bbox_batch += loss_bbox / self.batch_size / num_dec_layers
                        loss_giou_batch += loss_giou / self.batch_size / num_dec_layers

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5) #0.1
                self.optimizer.step()
                #self.scheduler.step_update(step)
                losses.append(loss.item())
                class_losses.append(loss_class_batch.item())
                box_losses.append(loss_bbox_batch.item())
                giou_losses.append(loss_giou_batch.item())
                step+=1
            
            print(f"Batch Loss: {loss.item():.4f}, Class Loss: {loss_class_batch.item():.4f}, "
                f"BBox Loss: {loss_bbox_batch.item():.4f}, GIoU Loss: {loss_giou_batch.item():.4f}")

            self.log_epoch_losses(epoch, np.array(losses), np.array(class_losses),np.array(box_losses), np.array(giou_losses))
            self.save_checkpoint(epoch)
            # ✅ Save "best model" based on minimal loss
            #avg_loss = np.mean(losses)
            avg_loss = loss.item()
            if avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_best.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()
                    },
                    ckpt_path,
                )
                print(f"🔥 New best model saved (epoch {epoch+1})")
                print(f"step value: {step}")
            torch.cuda.empty_cache()
            import gc; gc.collect()
