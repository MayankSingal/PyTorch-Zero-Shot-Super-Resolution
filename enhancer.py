import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision
import numpy as np
import cv2
import random
import net
import numpy
from torchvision import transforms
from utils import *
import matplotlib.image as img



def init_weights(m):
	
	if type(m) == nn.modules.conv.Conv2d:
		print("Weights initialized for:", m)
		torch.nn.init.xavier_uniform(m.weight)
		m.bias.data.fill_(0.01)


def enhance(img_path, scale):

	SRNet = net.SRNet().cuda()
	SRNet.apply(init_weights)

	criterion = nn.L1Loss().cuda()

	optimizer = torch.optim.Adam(SRNet.parameters(), lr=0.001)

	SRNet.train()

	image = img.imread(img_path)
	hr_fathers_sources = [image]

	scale_factors = np.array([[1.0, 1.5], [1.5, 1.0], [1.5, 1.5], [1.5, 2.0], [2.0, 1.5], [2.0, 2.0]])
	back_projection_iters = np.array([6, 6, 8, 10, 10, 12])
	learning_rate_change_iter_nums = [0]

	rec_mse = []
	steps_mse = []

	
	for sf_ind, scale in enumerate(scale_factors):

		for i in range(10000):

			hr_father = random_augment(ims=hr_fathers_sources,
									   base_scales = [1.0] + list(scale_factors),
									   leave_as_is_probability=0.05,
									   no_interpolate_probability=0.45,
									   min_scale=0.5,
									   max_scale=([1.0]+list(scale_factors))[len(hr_fathers_sources)-1],
									   allow_rotation=True,
									   scale_diff_sigma=0.25,
									   shear_sigma=0.1,
									   crop_size=128
									   )

			lr_son = father_to_son(hr_father, scale)
			lr_son_interpolated = imresize(lr_son, scale, hr_father.shape, "cubic")

			hr_father = torch.from_numpy(hr_father).unsqueeze(0).cuda().permute(0,3,1,2).float()
			lr_son_interpolated = torch.from_numpy(lr_son_interpolated).unsqueeze(0).cuda().permute(0,3,1,2).float()

			sr_son = SRNet(lr_son_interpolated)

			loss = criterion(sr_son, hr_father)

			if(not i % 50):
				son_out = father_to_son(image, scale)
				son_out_inter = imresize(son_out, scale, image.shape, "cubic")
				son_out_inter = torch.from_numpy(son_out_inter).unsqueeze(0).cuda().permute(0,3,1,2).float()				
				sr_son_out = SRNet(son_out_inter).permute(0,2,3,1).squeeze().data.cpu().numpy()
				sr_son_out = np.clip(np.squeeze(sr_son_out), 0, 1)
				rec_mse.append(np.mean(np.ndarray.flatten(np.square(image - sr_son_out))))
				steps_mse.append(i)

			lr_policy(i, optimizer, learning_rate_change_iter_nums, steps_mse, rec_mse)

			#curr_lr = 100
			for param_group in optimizer.param_groups:
				#if param_group['lr'] < 9e-6:
				curr_lr = param_group['lr']
				break




			optimizer.zero_grad()
			loss.backward()
			optimizer.step()		

			if i%10 == 0:
				print("Iteration:", i, "Loss:",loss.item())

			if curr_lr < 9e-6:
				break
		

	    ### Evaluation the result

		lr_img = img.imread(img_path)
		
		interpolated_lr_img = imresize(lr_img, scale, None, "cubic")
		interpolated_lr_img = torch.from_numpy(interpolated_lr_img).unsqueeze(0).cuda().permute(0,3,1,2).float()
		
		sr_img = infer(lr_img, scale, sf_ind, SRNet, back_projection_iters) #SRNet(interpolated_lr_img)

		save_img = torch.from_numpy(sr_img).unsqueeze(0).permute(0,3,1,2)
		torchvision.utils.save_image((save_img),img_path.split(".")[0]+'SR.'+ img_path.split(".")[1], normalize=False)
		torchvision.utils.save_image((interpolated_lr_img),img_path.split(".")[0]+'LR.'+img_path.split(".")[1] , normalize=False)

		hr_fathers_sources.append(sr_img)
		print("Optimization done for scale", scale)



def infer(input_img, scale, sf_ind, SRNet, back_projection_iters):
	
	outputs = []

	for k in range(0, 1+7, 1+int(scale[0] != scale[1])):
		test_img = np.rot90(input_img, k) if k < 4 else np.fliplr(np.rot90(input_img,k))
		interpolated_test_img = imresize(test_img, scale, None, "cubic")
		interpolated_test_img = torch.from_numpy(interpolated_test_img).unsqueeze(0).cuda().permute(0,3,1,2).float()
		tmp_output = SRNet(interpolated_test_img)
		tmp_output = tmp_output.permute(0,2,3,1).squeeze().data.cpu().numpy()
		tmp_output = np.clip(np.squeeze(tmp_output), 0, 1)

		tmp_output = np.rot90(tmp_output, -k) if k < 4 else np.rot90(np.fliplr(tmp_output), k)

		for bp_iter in range(back_projection_iters[sf_ind]):
			tmp_output = back_projection(tmp_output, input_img, "cubic", "cubic", scale)

		outputs.append(tmp_output)


	outputs_pre = np.median(outputs, 0)

	for bp_iter in range(back_projection_iters[sf_ind]):
		outputs_pre = back_projection(outputs_pre, input_img, "cubic", "cubic", scale)

	return outputs_pre


def lr_policy(iters, optimizer, learning_rate_change_iter_nums, mse_steps, mse_rec):

	if ((not (1 + iters) % 60) and (iters - learning_rate_change_iter_nums[-1] > 256)):
		[slope, _], [[var,_],_] = np.polyfit(mse_steps[(-256//50):], mse_rec[(-256//50):], 1, cov=True)

		std = np.sqrt(var)

		print('Slope:', slope, "STD:", std)

		if -1.5*slope < std:
			for param_group in optimizer.param_groups:
				param_group['lr'] = param_group['lr'] * 0.8
			print("Learning Rate Updated:", param_group['lr'])
			learning_rate_change_iter_nums.append(iters)
		











if __name__ == '__main__':
	## First argument is the image that you want to upsample with ZSSR. 
        ## Second argument is the scale with which you want to resize. Currently only scale = 2 supported. For other scales, change the variable 'scale_factors' accordingly.

	enhance('images/JFK.png', 2)

