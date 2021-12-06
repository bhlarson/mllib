import torch
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def model_weights(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
    return total_params

# compute network parameter stats
def model_stats(model):
    weight_array = None
    bias_array = None
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()

        array = torch.flatten(parameter)

        if name.split('.')[-1] == 'weight':
            if weight_array is None:
                weight_array = array
            else:
                weight_array = torch.cat((weight_array, array))

        elif name.split('.')[-1] == 'bias':
            if bias_array is None:
                bias_array = array
            else:
                bias_array = torch.cat((bias_array, array))

    weight_std, weight_mean = torch.std_mean(weight_array, unbiased=False)
    bias_std, bias_mean = torch.std_mean(bias_array, unbiased=False)

    return weight_std, weight_mean, bias_std, bias_mean