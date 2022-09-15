import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import torch
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import numpy as np

sns.set_style('whitegrid')
train_G_loss = []
train_D_loss = []
x = []

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

print(data_loader)
print(dataset)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    tmp1 = 0
    tmp2 = 0
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        errors = model.get_current_errors()
        tmp1 += errors['G_GAN'] + errors['G_L1']
        tmp2 += (errors['D_real'] + errors['D_fake']) * 0.5
        if total_steps % opt.print_freq == 0:
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    train_G_loss.append(tmp1 / len(dataset))
    train_D_loss.append(tmp2 / len(dataset))
    x.append(epoch)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()

    print(epoch, " : ", train_G_loss[-1], " , ", train_D_loss[-1])

color = cm.viridis(0.5)
f, ax = plt.subplots(1, 1)
ax.plot(x, train_G_loss, color=color)
ax.set_xlabel('epoch')
ax.set_ylabel('train_G_loss')
# ax.legend()
exp_dir = 'Plot/'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
else:
    os.makedirs(exp_dir, exist_ok=True)
filename = 'train_Gloss_3e-4'
f.savefig(os.path.join('Plot', filename + '.png'), dpi=1000)
np.savetxt(os.path.join('Plot', filename + '.txt'), train_G_loss)

f, ax = plt.subplots(1, 1)
ax.plot(x, train_G_loss, color=color)
ax.set_xlabel('epoch')
ax.set_ylabel('train_D_loss')
filename = 'train_Dloss_3e-4'
f.savefig(os.path.join('Plot', filename + '.png'), dpi=1000)
np.savetxt(os.path.join('Plot', filename + '.txt'), train_D_loss)
plt.show()
