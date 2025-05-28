import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, jaccard_score, confusion_matrix
import torch
import torch.nn.functional as F
import time
from datetime import datetime
import json

class MetricsTracker:
    """
    Class for tracking and saving model performance metrics during training
    and validation. Supports multiple metrics and epoch-wise tracking.
    """
    def __init__(self, save_dir='metrics', experiment_name=None):
        """
        Initialize metrics tracker.
        
        Args:
            save_dir: Directory to save metrics data
            experiment_name: Name for this experiment. If None, uses timestamp
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set experiment name (default: timestamp)
        if experiment_name is None:
            self.experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.experiment_name = experiment_name
            
        # Initialize metrics dictionary
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'dice_score': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'iou': [],
            'accuracy': [],
            'learning_rate': [],
            'time_per_epoch': []
        }
        
        # For storing best metrics
        self.best_metrics = {
            'best_dice': 0.0,
            'best_epoch': 0,
            'best_loss': float('inf')
        }
        
        # Validation prediction samples (for visualization)
        self.val_samples = []
        
        # Start time for epoch tracking
        self.epoch_start_time = None
        
    def start_epoch(self):
        """Mark the start of an epoch for timing purposes"""
        self.epoch_start_time = time.time()
        
    def end_epoch(self, epoch, train_loss, val_loss, val_preds, val_targets, learning_rate):
        """
        Record metrics at the end of an epoch
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
            val_preds: Model predictions on validation set (tensor)
            val_targets: Ground truth for validation set (tensor)
            learning_rate: Current learning rate
        """
        # Calculate time taken for this epoch
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
        else:
            epoch_time = 0
            
        # Convert tensors to numpy arrays for metric calculation
        if torch.is_tensor(val_preds):
            val_preds = val_preds.detach().cpu().numpy()
        if torch.is_tensor(val_targets):
            val_targets = val_targets.detach().cpu().numpy()
            
        # Binarize predictions
        binary_preds = (val_preds > 0.5).astype(np.float32)
        binary_targets = (val_targets > 0.5).astype(np.float32)
        
        # Flatten arrays for metric calculation
        flat_preds = binary_preds.reshape(-1)
        flat_targets = binary_targets.reshape(-1)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            flat_targets, flat_preds, average='binary', zero_division=0
        )
        
        # IoU (Jaccard index)
        iou = jaccard_score(flat_targets, flat_preds, average='binary', zero_division=0)
        
        # Accuracy
        correct = (flat_preds == flat_targets).sum()
        accuracy = correct / flat_targets.size
        
        # Dice coefficient
        dice = 2 * (flat_preds * flat_targets).sum() / (flat_preds.sum() + flat_targets.sum() + 1e-6)
        
        # Update metrics dictionary
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['dice_score'].append(dice)
        self.metrics['precision'].append(precision)
        self.metrics['recall'].append(recall)
        self.metrics['f1_score'].append(f1)
        self.metrics['iou'].append(iou)
        self.metrics['accuracy'].append(accuracy)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['time_per_epoch'].append(epoch_time)
        
        # Update best metrics if needed
        if dice > self.best_metrics['best_dice']:
            self.best_metrics['best_dice'] = dice
            self.best_metrics['best_epoch'] = epoch
            
        if val_loss < self.best_metrics['best_loss']:
            self.best_metrics['best_loss'] = val_loss
            
        # Store a few sample predictions for visualization
        # We'll keep 3 samples (first, middle, last)
        n_samples = binary_targets.shape[0]
        if n_samples > 0:
            indices = [0, n_samples // 2, n_samples - 1]
            indices = [i for i in indices if i < n_samples]
            
            for idx in indices:
                if idx < len(binary_preds):
                    self.val_samples.append({
                        'epoch': epoch,
                        'pred': binary_preds[idx],
                        'target': binary_targets[idx]
                    })
        
        # Save metrics after each epoch
        self.save_metrics()
        
        return dice  # Return dice score for convenience
    
    def save_metrics(self):
        """Save metrics to CSV and JSON files"""
        # Save to CSV
        metrics_df = pd.DataFrame(self.metrics)
        metrics_df.to_csv(os.path.join(self.save_dir, f"{self.experiment_name}_metrics.csv"), index=False)
        
        # Save best metrics to JSON
        with open(os.path.join(self.save_dir, f"{self.experiment_name}_best_metrics.json"), 'w') as f:
            json.dump(self.best_metrics, f, indent=4)
            
        # Save sample predictions (optional, can be memory intensive)
        # np.save(os.path.join(self.save_dir, f"{self.experiment_name}_samples.npy"), self.val_samples)
        
    def plot_metrics(self, save_plot=True):
        """Plot training and validation metrics"""
        if len(self.metrics['epoch']) == 0:
            print("No metrics to plot yet")
            return
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss
        axes[0, 0].plot(self.metrics['epoch'], self.metrics['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.metrics['epoch'], self.metrics['val_loss'], label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot Dice score
        axes[0, 1].plot(self.metrics['epoch'], self.metrics['dice_score'], label='Dice Score')
        axes[0, 1].plot(self.metrics['epoch'], self.metrics['iou'], label='IoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Dice Score and IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision, recall, F1
        axes[1, 0].plot(self.metrics['epoch'], self.metrics['precision'], label='Precision')
        axes[1, 0].plot(self.metrics['epoch'], self.metrics['recall'], label='Recall')
        axes[1, 0].plot(self.metrics['epoch'], self.metrics['f1_score'], label='F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision, Recall, and F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot accuracy and learning rate
        ax1 = axes[1, 1]
        ax1.plot(self.metrics['epoch'], self.metrics['accuracy'], 'b-', label='Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        ax2.plot(self.metrics['epoch'], self.metrics['learning_rate'], 'r-', label='Learning Rate')
        ax2.set_ylabel('Learning Rate', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax1.set_title('Accuracy and Learning Rate')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.save_dir, f"{self.experiment_name}_metrics_plot.png"))
            plt.close()
        else:
            plt.show()
            
    def plot_confusion_matrix(self, predictions, targets, save_plot=True):
        """Plot confusion matrix for binary segmentation"""
        # Convert to binary
        if torch.is_tensor(predictions):
            predictions = (predictions > 0.5).detach().cpu().numpy()
        else:
            predictions = (predictions > 0.5).astype(np.float32)
            
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
            
        # Flatten
        flat_preds = predictions.flatten()
        flat_targets = targets.flatten()
        
        # Compute confusion matrix
        cm = confusion_matrix(flat_targets, flat_preds)
        
        # Normalize to get percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Raw counts
        ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.set_title('Confusion Matrix (Counts)')
        tick_marks = np.arange(2)
        ax1.set_xticks(tick_marks)
        ax1.set_yticks(tick_marks)
        ax1.set_xticklabels(['Negative', 'Positive'])
        ax1.set_yticklabels(['Negative', 'Positive'])
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Add counts
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # Normalized percentages
        ax2.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xticks(tick_marks)
        ax2.set_yticks(tick_marks)
        ax2.set_xticklabels(['Negative', 'Positive'])
        ax2.set_yticklabels(['Negative', 'Positive'])
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        # Add percentages
        thresh = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                ax2.text(j, i, format(cm_normalized[i, j], '.2f'),
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.save_dir, f"{self.experiment_name}_confusion_matrix.png"))
            plt.close()
        else:
            plt.show()
            
    def get_current_metrics_summary(self):
        """Get a summary of current metrics as a dictionary"""
        if not self.metrics['epoch']:
            return {"status": "No metrics recorded yet"}
            
        latest_idx = -1
        
        summary = {
            "current_epoch": self.metrics['epoch'][latest_idx],
            "train_loss": self.metrics['train_loss'][latest_idx],
            "val_loss": self.metrics['val_loss'][latest_idx],
            "dice_score": self.metrics['dice_score'][latest_idx],
            "precision": self.metrics['precision'][latest_idx],
            "recall": self.metrics['recall'][latest_idx],
            "f1_score": self.metrics['f1_score'][latest_idx],
            "iou": self.metrics['iou'][latest_idx],
            "accuracy": self.metrics['accuracy'][latest_idx],
            "learning_rate": self.metrics['learning_rate'][latest_idx],
            "time_per_epoch": self.metrics['time_per_epoch'][latest_idx],
            "best_dice": self.best_metrics['best_dice'],
            "best_epoch": self.best_metrics['best_epoch'],
            "best_loss": self.best_metrics['best_loss']
        }
        
        return summary
    
    def compare_experiments(self, other_csv_paths, metrics_to_compare=None):
        """
        Compare this experiment with other experiments
        
        Args:
            other_csv_paths: List of paths to other experiments' CSV files
            metrics_to_compare: List of metrics to compare (default: dice_score and val_loss)
        """
        if metrics_to_compare is None:
            metrics_to_compare = ['dice_score', 'val_loss']
            
        # Load current experiment data
        current_df = pd.DataFrame(self.metrics)
        
        # Load other experiments
        other_dfs = []
        experiment_names = []
        
        for path in other_csv_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    other_dfs.append(df)
                    # Extract experiment name from filename
                    exp_name = os.path.basename(path).split('_metrics.csv')[0]
                    experiment_names.append(exp_name)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
        
        if not other_dfs:
            print("No other experiments to compare with")
            return
            
        # Plot comparisons
        for metric in metrics_to_compare:
            if metric in current_df.columns:
                plt.figure(figsize=(10, 6))
                
                # Plot current experiment
                plt.plot(current_df['epoch'], current_df[metric], 
                         label=f"{self.experiment_name}", linewidth=2)
                
                # Plot other experiments
                for i, df in enumerate(other_dfs):
                    if metric in df.columns:
                        plt.plot(df['epoch'], df[metric], 
                                 label=f"{experiment_names[i]}", linestyle='--')
                
                plt.xlabel('Epoch')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.title(f'Comparison of {metric.replace("_", " ").title()} Across Experiments')
                plt.legend()
                plt.grid(True)
                
                # Save plot
                plt.savefig(os.path.join(self.save_dir, f"comparison_{metric}.png"))
                plt.close()
            else:
                print(f"Metric {metric} not found in current experiment")

# Functions for computing additional metrics
def compute_pixel_accuracy(pred, target):
    """Compute pixel-wise accuracy"""
    if torch.is_tensor(pred):
        pred = (pred > 0.5).float()
    if torch.is_tensor(target):
        target = target.float()
        
    correct = (pred == target).sum()
    total = target.numel()
    return correct.float() / total

def compute_iou(pred, target, smooth=1e-6):
    """Compute Intersection over Union (Jaccard Index)"""
    if torch.is_tensor(pred):
        pred = (pred > 0.5).float()
        target = target.float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou
    else:
        pred = (pred > 0.5).astype(np.float32)
        target = target.astype(np.float32)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou

def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient for evaluation"""
    if torch.is_tensor(pred):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice
    else:
        pred = pred.flatten()
        target = target.flatten()
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice

def evaluate_thresholds(model, val_loader, device, thresholds=None):
    """
    Evaluate model performance across different thresholds
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run evaluation on
        thresholds: List of thresholds to evaluate (default: 0.1 to 0.9 in 0.1 increments)
    
    Returns:
        DataFrame with metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
        
    results = []
    
    model.eval()
    
    with torch.no_grad():
        all_preds = []
        all_targets = []
        
        # Collect all predictions and targets
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            all_preds.append(output.cpu())
            all_targets.append(target.cpu())
            
        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Evaluate at different thresholds
        for threshold in thresholds:
            binary_preds = (all_preds > threshold).float()
            
            # Compute metrics
            dice = dice_coefficient(binary_preds, all_targets)
            iou = compute_iou(binary_preds, all_targets)
            accuracy = compute_pixel_accuracy(binary_preds, all_targets)
            
            # Compute precision, recall, F1
            flat_preds = binary_preds.view(-1).cpu().numpy()
            flat_targets = all_targets.view(-1).cpu().numpy()
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                flat_targets, flat_preds, average='binary', zero_division=0
            )
            
            results.append({
                'threshold': threshold,
                'dice': dice.item(),
                'iou': iou.item(),
                'accuracy': accuracy.item(),
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def plot_threshold_analysis(results_df, save_path=None):
    """Plot metrics across different thresholds"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(results_df['threshold'], results_df['dice'], label='Dice', marker='o')
    plt.plot(results_df['threshold'], results_df['iou'], label='IoU', marker='s')
    plt.plot(results_df['threshold'], results_df['precision'], label='Precision', marker='^')
    plt.plot(results_df['threshold'], results_df['recall'], label='Recall', marker='v')
    plt.plot(results_df['threshold'], results_df['f1'], label='F1', marker='D')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        
    # Find optimal thresholds
    best_dice_idx = results_df['dice'].idxmax()
    best_iou_idx = results_df['iou'].idxmax()
    best_f1_idx = results_df['f1'].idxmax()
    
    print(f"Optimal threshold for Dice: {results_df.iloc[best_dice_idx]['threshold']:.2f} (Dice = {results_df.iloc[best_dice_idx]['dice']:.4f})")
    print(f"Optimal threshold for IoU: {results_df.iloc[best_iou_idx]['threshold']:.2f} (IoU = {results_df.iloc[best_iou_idx]['iou']:.4f})")
    print(f"Optimal threshold for F1: {results_df.iloc[best_f1_idx]['threshold']:.2f} (F1 = {results_df.iloc[best_f1_idx]['f1']:.4f})")
    
    return {
        'best_dice_threshold': results_df.iloc[best_dice_idx]['threshold'],
        'best_iou_threshold': results_df.iloc[best_iou_idx]['threshold'],
        'best_f1_threshold': results_df.iloc[best_f1_idx]['threshold']
    } 