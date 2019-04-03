import os

import torch


def torch_save_model(model, file_prefix=None, file_name=None, val_loss='None', best_loss='None', enforcement=False):
    # Save model
    try:
        if enforcement or val_loss == 'None' or best_loss == 'None':
            file_path = '{}{}_{}.pkl'.format(file_prefix, file_name, 'enforcement')
            torch.save(model, file_path)
            print('enforcement save:', file_path)

        elif val_loss != 'None' and best_loss != 'None' and ~enforcement:
            is_best = val_loss < best_loss
            best_loss = min(best_loss, val_loss)
            if is_best:
                file_path = '{}{}_{:.4f}.pkl'.format(file_prefix, file_name, best_loss)
                torch.save(model, file_path)
            return best_loss
    except Exception as e:
        # if error, save model in default path
        print(e)
        file_path = '{}{}.pkl'.format(os.getcwd(), '/default')
        print('default save:', file_path)
        torch.save(model, file_path)