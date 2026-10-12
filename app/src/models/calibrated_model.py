"""
Module: calibrated_model.py
Description: Probability Calibration and Loss Functions for Depression Severity Detection.
Includes:
- FocalLoss: Class-weighted multi-class Focal Loss.
- TemperatureScaler: Post-hoc Platt scaling calibration for multi-class logits.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class FocalLoss(nn.Module):
        """
        Focal Loss for Multi-Class Classification with Class Weighting.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        Reduces relative loss for easy-to-classify examples and focuses training on minority categories.
        """
        def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
            super().__init__()
            self.alpha = alpha  # Tensor of weights per class shape (num_classes,)
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, logits, targets):
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss

            if self.alpha is not None:
                if self.alpha.device != logits.device:
                    self.alpha = self.alpha.to(logits.device)
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            return focal_loss


    class TemperatureScaler(nn.Module):
        """
        Temperature Scaling Calibration (Platt Scaling Extension for Multi-Class).
        Optimizes single scalar T > 0 on validation set using Negative Log-Likelihood (NLL).
        Calibrates unscaled logits: p_i = softmax(logits / T).
        Solves network overconfidence.
        """
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        def forward(self, *args, **kwargs):
            outputs = self.model(*args, **kwargs)
            logits = outputs['severity_logits']
            scaled_logits = logits / self.temperature
            outputs['severity_logits_unscaled'] = logits
            outputs['severity_logits'] = scaled_logits
            outputs['severity_probs'] = F.softmax(scaled_logits, dim=-1)
            outputs['temperature'] = self.temperature.item()
            return outputs

        def calibrate(self, val_loader, device='cpu'):
            """
            Fits temperature T on validation set using L-BFGS optimizer.
            """
            self.model.eval()
            self.to(device)

            nll_criterion = nn.CrossEntropyLoss()
            logits_list = []
            labels_list = []

            with torch.no_grad():
                for batch in val_loader:
                    if 'x_fused' in batch:
                        out = self.model(batch['x_fused'].to(device))
                    else:
                        e = batch['egemaps'].to(device)
                        t = batch.get('temporal', None)
                        if t is not None: t = t.to(device)
                        w = batch['wav2vec'].to(device)
                        txt = batch.get('text_bert', None)
                        if txt is not None: txt = txt.to(device)
                        out = self.model(e, t, w, txt)

                    y = batch['class_id'].to(device)
                    logits_list.append(out['severity_logits'])
                    labels_list.append(y)

            logits = torch.cat(logits_list, dim=0)
            labels = torch.cat(labels_list, dim=0)

            optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

            def eval_fn():
                optimizer.zero_grad()
                loss = nll_criterion(logits / self.temperature, labels)
                loss.backward()
                return loss

            optimizer.step(eval_fn)
            print(f"[CALIBRATION COMPLETE] Fitted Optimal Temperature T = {self.temperature.item():.4f}")
            return self.temperature.item()

else:
    class FocalLoss:
        pass
    class TemperatureScaler:
        pass
