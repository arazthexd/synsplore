import torch
import torch.nn.functional as F
import torchmetrics

def dice_loss(pred, target, smooth=1):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W).
        target: Tensor of ground truth (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)
    
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    
    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Return Dice Loss
    return dice.mean()

def acc(pred, target, **kwargs):
    """
    Computes the accuracy metric for classification tasks.
    Args:
        pred: Tensor of model predictions or logits.
        target: Tensor of ground truth labels.
        **kwargs: Additional keyword arguments passed to torchmetrics.Accuracy.
    Returns:
        Accuracy score as a tensor.
    """
    return torchmetrics.Accuracy(**kwargs)(pred, target)

def roc_auc(pred, target, **kwargs):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (ROC-AUC).
    Args:
        pred: Tensor of model predictions or probabilities.
        target: Tensor of ground truth binary labels.
        **kwargs: Additional keyword arguments passed to torchmetrics.AUROC.
    Returns:
        ROC-AUC score as a tensor.
    """
    return torchmetrics.AUROC(**kwargs)(pred, target)

def _extract_tp_tn_fp_fn(confusion_matrix, num_classes):
    """
    Extracts True Positive, True Negative, False Positive, and False Negative
    values for each class from a multi-class confusion matrix.
    
    Args:
        confusion_matrix: Tensor of shape (num_classes, num_classes) representing
                         the confusion matrix where element [i,j] represents 
                         samples with true label i predicted as label j.
        num_classes: Integer representing the number of classes.
    
    Returns:
        Dictionary containing:
            - 'tp': Tensor of shape (num_classes,) with TP values for each class
            - 'tn': Tensor of shape (num_classes,) with TN values for each class  
            - 'fp': Tensor of shape (num_classes,) with FP values for each class
            - 'fn': Tensor of shape (num_classes,) with FN values for each class
    """
    # Initialize tensors to store results
    tp = torch.zeros(num_classes, dtype=confusion_matrix.dtype, device=confusion_matrix.device)
    tn = torch.zeros(num_classes, dtype=confusion_matrix.dtype, device=confusion_matrix.device)
    fp = torch.zeros(num_classes, dtype=confusion_matrix.dtype, device=confusion_matrix.device)
    fn = torch.zeros(num_classes, dtype=confusion_matrix.dtype, device=confusion_matrix.device)
    
    # Calculate total sum of confusion matrix
    total_sum = confusion_matrix.sum()
    
    for i in range(num_classes):
        # True Positives: diagonal element for class i
        tp[i] = confusion_matrix[i, i]
        
        # False Negatives: sum of row i excluding diagonal
        fn[i] = confusion_matrix[i, :].sum() - confusion_matrix[i, i]
        
        # False Positives: sum of column i excluding diagonal
        fp[i] = confusion_matrix[:, i].sum() - confusion_matrix[i, i]
        
        # True Negatives: total - tp - fn - fp
        tn[i] = total_sum - tp[i] - fn[i] - fp[i]
    
    return {
        'tp': tp,
        'tn': tn, 
        'fp': fp,
        'fn': fn
    }

def confusion_matrix(pred, target, **kwargs):
    """
    Computes the confusion matrix for classification tasks.
    Args:
        pred: Tensor of model predictions or logits.
        target: Tensor of ground truth labels.
        **kwargs: Additional keyword arguments passed to torchmetrics.ConfusionMatrix.
    Returns:
        Confusion matrix as a tensor of shape (num_classes, num_classes).
    """
    cm = torchmetrics.ConfusionMatrix(**kwargs)
    return cm(pred, target)

class ConfusionMatrix(torchmetrics.ConfusionMatrix):
    def __call__(pred, target):
        matrix = super().__call__(pred, target)
        return matrix.flatten()