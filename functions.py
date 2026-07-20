import torch
import torch.nn as nn

class NLLloss(nn.Module):
    def __init__(self, reduction='both'):
        super().__init__()
        assert reduction in ('NLL','MSE','both')
        self.reduction = reduction

    def forward(self, mu,targets,mask, flux_errors = None,log_var = None):

        # Only compute loss at masked positions

        if log_var is not None:

            log_var = log_var.squeeze()
            mu = mu.squeeze()

            mse = (targets - mu).pow(2)[mask].mean()

            if flux_errors is not None:
                var_total = log_var.exp() + flux_errors.pow(2)
            else:
                var_total = log_var.exp()
            #var_total = flux_errors.pow(2)
            nll = 0.5 * ((targets - mu).pow(2) / var_total + var_total.log())

            if self.reduction == 'NLL':

                return nll[mask].mean()

            elif self.reduction == 'MSE':

                return mse

            else:
                return nll[mask].mean(), mse

        else:

            mu = mu.squeeze()
            mse = (targets - mu).pow(2)[mask].mean()

            if flux_errors is not None:
                var_total = flux_errors.pow(2)
                var_total = var_total.clamp(min=1e-6)
                nll = 0.5 * ((targets - mu).pow(2) / var_total + var_total.log())

            else:
                nll = 0.5 * ((targets - mu).pow(2))

            if self.reduction == 'NLL':

                return nll[mask].mean()

            elif self.reduction == 'MSE':

                return mse

            else:
                return nll[mask].mean(), mse
