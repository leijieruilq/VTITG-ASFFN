import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, device="cuda", dtype=None):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [batch_size] + list(o.size())[1:] for o in output
                ]

            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(batch_size, *in_size, device=device, dtype=dtype) for in_size in input_size]
    # print(type(x[0]))
    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)
    model.to(device=device, dtype=dtype)
    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()
    model_info =''
    # print("----------------------------------------------------------------")
    model_info += "----------------------------------------------------------------"
    model_info += '\n'
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    # print(line_new)
    model_info += line_new
    model_info += '\n'
    # print("================================================================")
    model_info += "================================================================"
    model_info += '\n'
    total_params = 0
    total_output = 0
    trainable_params = 0


    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        # print(line_new)
        model_info += line_new
        model_info += '\n'

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    
    # print("================================================================")
    model_info += "================================================================"
    model_info += '\n'
    # print("Total params: {0:,}".format(total_params))
    model_info += '{:<35}  {:>10} {:>15}'.format("Total params:",'',total_params)
    model_info += '\n'
    # print("Trainable params: {0:,}".format(trainable_params))
    model_info += '{:<35}  {:>10} {:>15}'.format("Trainable params:",'',trainable_params)
    model_info += '\n'
    # print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    model_info += '{:<35}  {:>10} {:>15}'.format("Non-trainable params:",'',total_params - trainable_params)
    model_info += '\n'
    # print("----------------------------------------------------------------")
    model_info += "----------------------------------------------------------------"
    # model_info += '\n'
    # # print("Input size (MB): %0.2f" % total_input_size)
    # model_info += '{:<30}  {:>10} {:>15} (MB)'.format("Input size:",'',"%0.2f"%total_input_size)
    # model_info += '\n'
    # # print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    # model_info += '{:<30}  {:>10} {:>15} (MB)'.format("Forward/backward pass size:",'',"%0.2f"%total_output_size)
    # model_info += '\n'
    # # print("Params size (MB): %0.2f" % total_params_size)
    # model_info += '{:<30}  {:>10} {:>15} (MB)'.format("Params size:",'',"%0.2f"%total_params_size)
    # model_info += '\n'
    # # print("Estimated Total Size (MB): %0.2f" % total_size)
    # model_info += '{:<30}  {:>10} {:>15} (MB)'.format("Estimated Total Size:",'',"%0.2f"%total_size)
    # model_info += '\n'
    # # print("----------------------------------------------------------------")
    # model_info += "----------------------------------------------------------------"
    # model_info += '\n'

    return model_info


def prod(obj):
    sum_num = 0
    for item in obj:
        if not type(item) == int:
            sum_num += prod(item)
        else:
            sum_num += item
    return sum_num