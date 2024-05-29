import torch
import torch.nn as nn

# import flopth
from flopth import flopth
from model import DM2FNet, DM2FNet_woPhy
from model_improve import DM2FNet_new2, DM2FNet_woPhy_new
from thop import profile, clever_format
import time

models = [DM2FNet, DM2FNet_woPhy, DM2FNet_new2, DM2FNet_woPhy_new]
macs_list = []
params_list = []
time_list = []
input = torch.randn(1, 3, 640, 480)
input_batch = torch.randn(1, 3, 640, 480)
epochs = 100
for model in models:
    my_model = model()
    
    my_model = my_model.cuda()
    my_model.eval()
    input = input.cuda()
    input_batch = input_batch.cuda()
    macs, params = profile(my_model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    macs_list.append(macs)
    params_list.append(params)
    temp_time = 0
    for i in range(epochs):
        start = time.time()
        output = my_model(input)
        end = time.time()
        temp_time += (end - start)

    time_one = temp_time / epochs
    time_list.append(f"{time_one:.4f} s")
    

for i in range(len(models)):
    print(models[i])
    print(macs_list[i], params_list[i])
    print(time_list[i])
