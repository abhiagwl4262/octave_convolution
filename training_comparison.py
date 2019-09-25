import os
import matplotlib.pyplot as plt  
import numpy as np 
import argparse


def main(args):
	if (args.plateau):
		file_vanilla = '/mnt/octave_convolution/checkpoint_vanilla_conv_plateau/log_cifar10_resnet_vanilla_conv_.txt'
		file_octave  =  '/mnt/octave_convolution/checkpoint_cifar_resnet18_octave_conv_plateau/log_cifar10_resnet_octave_conv_.txt'

		validation_acc_vanilla = []
		validation_acc_octave  = []


		with open(file_vanilla, "r") as f:
			lines = f.readlines()
			for line in lines:
				splitted_array = line.split(' ')
				valid_acc  = splitted_array[-4]
				validation_acc_vanilla.append(float(valid_acc))


		with open(file_octave, "r") as f:
			lines = f.readlines()
			for line in lines:
				splitted_array = line.split(' ')
				valid_acc  = splitted_array[-4]
				validation_acc_octave.append(float(valid_acc))

		# x = np.linspace(0, 1, 150)
		# plt.yticks(np.arange(0, 100, step=5))
		fig = plt.figure()
		plt.plot(validation_acc_octave,'-b', label='Octave')
		plt.plot(validation_acc_vanilla,'-r', label='vanilla')
		plt.axis([0, 150, 0, 100])
		plt.xlabel('epochs')
		plt.ylabel('validation_accuracy')
		plt.title('resnet18_cifar10_comparison')
		plt.show()

	else:

		# file_vanilla = '/mnt/octave_convolution/checkpoint_cifar_resnet18_octave_conv_step_LR_2_small_alpha/log_cifar10_resnet_octave_conv_.txt'
		file_vanilla = '/mnt/octave_convolution/checkpoint_cifar_resnet18_vanilla_conv_step_LR_2/log_cifar10_resnet_vanilla_conv_.txt'
		file_octave  =  '/mnt/octave_convolution/checkpoint_cifar_resnet18_octave_conv_step_LR_2/log_cifar10_resnet_octave_conv_.txt'

		validation_acc_vanilla = []
		validation_acc_octave  = []


		with open(file_vanilla, "r") as f:
			lines = f.readlines()
			for line in lines:
				splitted_array = line.split(' ')
				valid_acc  = splitted_array[-4]
				validation_acc_vanilla.append(float(valid_acc))


		with open(file_octave, "r") as f:
			lines = f.readlines()
			for line in lines:
				splitted_array = line.split(' ')
				valid_acc  = splitted_array[-4]
				validation_acc_octave.append(float(valid_acc))

		# x = np.linspace(0, 1, 150)
		# plt.yticks(np.arange(0, 100, step=5))
		fig = plt.figure()
		plt.plot(validation_acc_octave,'-b', label='Octave')
		plt.plot(validation_acc_vanilla,'-r', label='vanilla')
		plt.axis([0, 150, 0, 100])
		plt.xlabel('epochs')
		plt.ylabel('validation_accuracy')
		plt.title('resnet18_cifar10_comparison')
		plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--plateau', dest='plateau', action='store_true', help='evaluate model on validation set')

    main(parser.parse_args())