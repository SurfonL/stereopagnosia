'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Mukund Mundhra <mukundmundhra@cs.ucla.edu>

If this code is useful to you, please consider citing the following paper:

A. Wong, M. Mundhra, and S. Soatto. Stereopagnosia: Fooling Stereo Networks with Adversarial Perturbations.
https://arxiv.org/pdf/2009.10142.pdf

@inproceedings{wong2021stereopagnosia,
  title={Stereopagnosia: Fooling Stereo Networks with Adversarial Perturbations},
  author={Wong, Alex and Mundhra, Mukund and Soatto, Stefano},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  year={2021}
}
'''
import sys, random
import numpy as np
from skimage.util import random_noise
import torch
import global_constants as settings
from TargetAttack import TargetPerturbations, TargetPerturbationsModel, Transformation


class PerturbationsModel(object):
    '''
    Adversarial perturbation model

    Args:
        perturb_method : str
            method to use to generate perturbations
        perturb_mode : str
            left, right, or both
        output_norm : float
            upper (infinity) norm of adversarial noise
        n_step : int
            number of steps to optimize perturbations
        learning_rate : float
            learning rate (alpha) to be used for optimizing perturbations
        momentum : float
            momemtum (mu) used for momentum iterative fast gradient sign method
        probability_diverse_input : float
            probability to use diverse input
        device : torch.device
            device to run optimization
    '''

    def __init__(self,
                 perturb_method=settings.PERTURB_METHOD,
                 perturb_mode=settings.PERTURB_MODE,
                 transform_method=None,
                 output_norm=settings.OUTPUT_NORM,
                 n_step=settings.N_STEP,
                 learning_rate=settings.LEARNING_RATE,
                 learning_schedule = None,
                 momentum=settings.MOMENTUM,
                 probability_diverse_input=settings.PROBABILITY_DIVERSE_INPUT,
                 device=torch.device(settings.CUDA)):

        self.__perturb_method = perturb_method
        self.__perturb_mode = perturb_mode
        self.__transform_method = transform_method
        self.__output_norm = output_norm
        self.__n_step = n_step
        self.__learning_rate = learning_rate
        self.__learning_schedule = learning_schedule
        self.__momentum = momentum
        self.__probability_diverse_input = probability_diverse_input

        self.__device = device

    def forward(self, stereo_model, image0, image1, ground_truth, output_norm = None):
        '''
        Applies perturbations to image and clamp

        Args:
            stereo_model : object
                stereo network
            image0 : tensor
                N x C x H x W RGB image
            image1 : tensor
                N x C x H x W RGB image
            ground_truth : tensor
                N x 1 x H x W ground truth disparity

        Returns:
            tensor : adversarial noise/perturbations for left image
            tensor : adversarial noise/perturbations for right image
            tensor : adversarially perturbed left image
            tensor : adversarially perturbed right image
        '''

        if self.__perturb_method == 'fgsm':
            noise0, noise1 = self.__fgsm(stereo_model, image0, image1, ground_truth)

        elif self.__perturb_method == 'ifgsm':
            noise0, noise1 = self.__ifgsm(stereo_model, image0, image1, ground_truth)

        elif self.__perturb_method == 'mifgsm':
            noise0, noise1 = self.__mifgsm(stereo_model, image0, image1, ground_truth)

        elif self.__perturb_method == 'gaussian':
            noise0, noise1 = self.__gaussian(image0, image1)

        elif self.__perturb_method == 'uniform':
            noise0, noise1 = self.__uniform(image0, image1)

        elif self.__perturb_method == 'target':
            noise0, noise1 = self.__target(stereo_model, image0, image1, ground_truth)

        elif self.__perturb_method == 'none':
            noise0 = torch.zeros_like(image0)
            noise1 = torch.zeros_like(image1)

        else:
            raise ValueError('Invalid perturbation method: %s' % self.__perturb_method)

        # Add to image and clamp between supports of image intensity to get perturbed image
        image0_output = torch.clamp(image0 + noise0, 0.0, 1.0)
        image1_output = torch.clamp(image1 + noise1, 0.0, 1.0)

        # Output perturbations are the difference between original and perturbed image
        noise0_output = image0_output - image0
        noise1_output = image1_output - image1

        return noise0_output, noise1_output, image0_output, image1_output

    def __gaussian(self, image0, image1):
        '''
        Computes gaussian noise as perturbations

        Args:
            image0 : tensor
                N x C x H x W RGB image
            image1 : tensor
                N x C x H x W RGB image

        Returns:
            tensor : gaussian noise/perturbations for left image
            tensor : gaussian noise/perturbations for right image
        '''

        variance = (self.__output_norm / 4.0) ** 2

        # Apply gaussian noise to images
        image0_output = random_noise(
            image0.clone().detach().cpu().numpy(),
            mode='gaussian',
            var=variance)

        image1_output = random_noise(
            image1.clone().detach().cpu().numpy(),
            mode='gaussian',
            var=variance)

        image0_output = torch.from_numpy(image0_output).float()
        image1_output = torch.from_numpy(image1_output).float()

        if self.__device.type == settings.CUDA:
            image0 = image0.cuda()
            image1 = image1.cuda()
            image0_output = image0_output.cuda()
            image1_output = image1_output.cuda()
        else:
            image0 = image0.cpu()
            image1 = image1.cpu()
            image0_output = image0_output.cpu()
            image1_output = image1_output.cpu()

        # Subtract perturbed images from original images to get noise
        if self.__perturb_mode == 'both':
            noise0_output = image0_output - image0
            noise1_output = image1_output - image1

        elif self.__perturb_mode == 'left':
            noise0_output = image0_output - image0
            noise1_output = torch.zeros_like(image1)

        elif self.__perturb_mode == 'right':
            noise0_output = torch.zeros_like(image0)
            noise1_output = image1_output - image1

        else:
            raise ValueError('Invalid perturbation mode: %s' % self.__perturb_mode)

        return noise0_output, noise1_output

    def __uniform(self, image0, image1):
        '''
        Computes uniform noise as pertubations

        Args:
            image0 : tensor
                N x C x H x W RGB image
            image1 : tensor
                N x C x H x W RGB image

        Returns:
            tensor : uniform noise/perturbations for left image
            tensor : uniform noise/perturbations for right image
        '''

        # Compute uniform noise to images
        noise0_output = np.random.uniform(
            size=image0.clone().detach().cpu().numpy().shape,
            low=-self.__output_norm,
            high=self.__output_norm)

        noise1_output = np.random.uniform(
            size=image1.clone().detach().cpu().numpy().shape,
            low=-self.__output_norm,
            high=self.__output_norm)

        if self.__perturb_mode == 'both':
            noise0_output = torch.from_numpy(noise0_output).float()
            noise1_output = torch.from_numpy(noise1_output).float()

        elif self.__perturb_mode == 'left':
            noise0_output = torch.from_numpy(noise0_output).float()
            noise1_output = torch.zeros_like(image1)

        elif self.__perturb_mode == 'right':
            noise0_output = torch.zeros_like(image0)
            noise1_output = torch.from_numpy(noise1_output).float()

        else:
            raise ValueError('Invalid perturbation mode: %s' % self.__perturb_mode)

        if self.__device.type == settings.CUDA:
            noise0_output = noise0_output.cuda()
            noise1_output = noise1_output.cuda()
        else:
            noise0_output = noise0_output.cpu()
            noise1_output = noise1_output.cpu()

        return noise0_output, noise1_output

    def __fgsm(self, stereo_model, image0, image1, ground_truth):
        '''
        Computes adversarial perturbations using fast gradient sign method

        Args:
            stereo_model : object
                stereo network
            image0 : tensor
                N x C x H x W RGB image
            image1 : tensor
                N x C x H x W RGB image
            ground_truth : tensor
                N x 1 x H x W ground truth disparity

        Returns:
            tensor : uniform noise/perturbations for left image
            tensor : uniform noise/perturbations for right image
        '''

        # Set gradients for image to be true
        image0 = torch.autograd.Variable(image0, requires_grad=True)
        image1 = torch.autograd.Variable(image1, requires_grad=True)

        # Compute loss
        loss = stereo_model.compute_loss(image0, image1, ground_truth)
        loss.backward(retain_graph=True)

        # Compute perturbations based on fast gradient sign method
        if self.__perturb_mode == 'both':
            noise0_output = self.__output_norm * torch.sign(image0.grad.data)
            noise1_output = self.__output_norm * torch.sign(image1.grad.data)

        elif self.__perturb_mode == 'left':
            noise0_output = self.__output_norm * torch.sign(image0.grad.data)
            noise1_output = torch.zeros_like(image1)

        elif self.__perturb_mode == 'right':
            noise0_output = torch.zeros_like(image0)
            noise1_output = self.__output_norm * torch.sign(image1.grad.data)

        else:
            raise ValueError('Invalid perturbation mode: %s' % self.__perturb_mode)

        return noise0_output, noise1_output

    def __ifgsm(self, stereo_model, image0, image1, ground_truth):
        '''
        Computes adversarial perturbations using iterative fast gradient sign method

        Args:
            stereo_model : object
                stereo network
            image0 : tensor
                N x C x H x W RGB image
            image1 : tensor
                N x C x H x W RGB image
            ground_truth : tensor
                N x 1 x H x W ground truth disparity

        Returns:
            tensor : uniform noise/perturbations for left image
            tensor : uniform noise/perturbations for right image
        '''

        image0_output = image0.clone()
        image1_output = image1.clone()

        for step in range(self.__n_step):

            # Set gradients for image to be true
            image0_output = torch.autograd.Variable(image0_output, requires_grad=True)
            image1_output = torch.autograd.Variable(image1_output, requires_grad=True)

            # Compute loss, input diversity is only used if probability greater than 0
            loss = stereo_model.compute_loss(
                *self.__diverse_input(image0_output, image1_output, ground_truth))
            loss.backward(retain_graph=True)

            # Compute perturbations using fast gradient sign method
            if self.__perturb_mode == 'both':
                noise0_output = self.__learning_rate * torch.sign(image0_output.grad.data)
                noise1_output = self.__learning_rate * torch.sign(image1_output.grad.data)

            elif self.__perturb_mode == 'left':
                noise0_output = self.__learning_rate * torch.sign(image0_output.grad.data)
                noise1_output = torch.zeros_like(image1)

            elif self.__perturb_mode == 'right':
                noise0_output = torch.zeros_like(image0)
                noise1_output = self.__learning_rate * torch.sign(image1_output.grad.data)

            else:
                raise ValueError('Invalid perturbation mode: %s' % self.__perturb_mode)

            # Add to image and clamp between supports of image intensity to get perturbed image
            image0_output = torch.clamp(image0_output + noise0_output, 0.0, 1.0)
            image1_output = torch.clamp(image1_output + noise1_output, 0.0, 1.0)

            # Output perturbations are the difference between original and perturbed image
            noise0_output = torch.clamp(image0_output - image0, -self.__output_norm, self.__output_norm)
            noise1_output = torch.clamp(image1_output - image1, -self.__output_norm, self.__output_norm)

            # Add perturbations to images
            image0_output = image0 + noise0_output
            image1_output = image1 + noise1_output

            sys.stdout.write(
                'Step={:3}/{:3}  Stereo Model Loss={:.10f}\r'.format(
                    step, self.__n_step, loss.item()))
            sys.stdout.flush()

        return noise0_output, noise1_output

    def __mifgsm(self, stereo_model, image0, image1, ground_truth):
        '''
        Computes adversarial perturbations using momentum iterative fast gradient sign method

        Args:
            stereo_model : object
                stereo network
            image0 : tensor
                N x C x H x W RGB image
            image1 : tensor
                N x C x H x W RGB image
            ground_truth : tensor
                N x 1 x H x W ground truth disparity

        Returns:
            tensor : uniform noise/perturbations for left image
            tensor : uniform noise/perturbations for right image
        '''

        image0_output = image0.clone()
        image1_output = image1.clone()

        grad0 = torch.zeros_like(image0)
        grad1 = torch.zeros_like(image1)

        for step in range(self.__n_step):

            # Set gradients for image to be true
            image0_output = torch.autograd.Variable(image0_output, requires_grad=True)
            image1_output = torch.autograd.Variable(image1_output, requires_grad=True)

            # Compute loss, input diversity is only used if probability greater than 0
            loss = stereo_model.compute_loss(
                *self.__diverse_input(image0_output, image1_output, ground_truth))
            loss.backward(retain_graph=True)

            # Compute gradients with momentum
            grad0 = self.__momentum * grad0 + \
                (1.0 - self.__momentum) * image0_output.grad.data / torch.sum(torch.abs(image0_output.grad.data))
            grad1 = self.__momentum * grad1 + \
                (1.0 - self.__momentum) * image1_output.grad.data / torch.sum(torch.abs(image1_output.grad.data))

            # Compute perturbations using fast gradient sign method
            if self.__perturb_mode == 'both':
                noise0_output = self.__learning_rate * torch.sign(grad0)
                noise1_output = self.__learning_rate * torch.sign(grad1)

            elif self.__perturb_mode == 'left':
                noise0_output = self.__learning_rate * torch.sign(grad0)
                noise1_output = torch.zeros_like(image1)

            elif self.__perturb_mode == 'right':
                noise0_output = torch.zeros_like(image0)
                noise1_output = self.__learning_rate * torch.sign(grad1)

            else:
                raise ValueError('Invalid perturbation mode: %s' % self.__perturb_mode)

            # Add to image and clamp between supports of image intensity to get perturbed image
            image0_output = torch.clamp(image0_output + noise0_output, 0.0, 1.0)
            image1_output = torch.clamp(image1_output + noise1_output, 0.0, 1.0)

            # Output perturbations are the difference between original and perturbed image
            noise0_output = torch.clamp(image0_output - image0, -self.__output_norm, self.__output_norm)
            noise1_output = torch.clamp(image1_output - image1, -self.__output_norm, self.__output_norm)

            # Add perturbations to images
            image0_output = image0 + noise0_output
            image1_output = image1 + noise1_output

            sys.stdout.write(
                'Step={:3}/{:3}  Stereo Model Loss={:.10f}\r'.format(
                    step, self.__n_step, loss.item()))
            sys.stdout.flush()

        return noise0_output, noise1_output

    def __diverse_input(self, image0, image1, ground_truth):

        # If p greater than probability of input diversity
        if torch.rand(1) > self.__probability_diverse_input:
            return image0, image1, ground_truth

        assert image0.shape == image1.shape

        # Compute padding on each side
        _, _, o_height, o_width = image0.shape

        n_height = random.randint(int(o_height - o_height / 10.0), o_height)
        n_width = random.randint(int(o_width - o_width / 10.0), o_width)

        top_pad = random.randint(0, o_height - n_height)
        bottom_pad = o_height - n_height - top_pad

        left_pad = random.randint(0, o_width - n_width)
        right_pad = o_width - n_width - left_pad

        # Resize images to new size and pad with zeros to get original size
        image0_resized = torch.nn.functional.interpolate(
            image0,
            size=(n_height, n_width),
            mode='bilinear')

        image0_output = torch.nn.functional.pad(
            image0_resized,
            pad=(left_pad, right_pad, top_pad, bottom_pad),
            mode='constant',
            value=0)

        image1_resized = torch.nn.functional.interpolate(
            image1,
            size=(n_height, n_width),
            mode='bilinear')

        image1_output = torch.nn.functional.pad(
            image1_resized,
            pad=(left_pad, right_pad, top_pad, bottom_pad),
            mode='constant',
            value=0)

        # Resize ground truth and pad with zeros
        ground_truth_resized = torch.nn.functional.interpolate(
            ground_truth,
            size=(n_height, n_width),
            mode='nearest')

        ground_truth_output = torch.nn.functional.pad(
            ground_truth_resized,
            pad=(left_pad, right_pad, top_pad, bottom_pad),
            mode='constant',
            value=0)

        # Scale disparity to adjust for change in size
        ground_truth_output *= float(n_width) / float(o_width)

        assert image0.shape == image0_output.shape
        assert image1.shape == image1_output.shape
        assert ground_truth.shape == ground_truth_output.shape

        return image0_output, image1_output, ground_truth_output

    def __target(self, stereo_model, image0, image1, ground_truth):
        logits = True


        b,c,h,w = image0.shape

        n_step = self.__n_step
        learning_rate = self.__learning_rate

        model = TargetPerturbationsModel(
            n_height=h,
            n_width=w,
            n_channel=3,
            output_norm = self.__output_norm
        )


        schedule_pos = 0
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate[0])

        # import neptune.new as neptune
        #
        # run = neptune.init(
        #     project="woojin.jeon337/Stereo-Target-Attack",
        #     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MjY5MDI4ZC03ZmQ0LTRhMDEtYmMyMC03MDZhNzYxZTQ4ODIifQ==",
        # )  # your credentials


        for step in range(1, n_step + 1):


            if step > self.__learning_schedule[schedule_pos]:
                schedule_pos = schedule_pos + 1
                learning_rate = self.__learning_rate[schedule_pos]

                # Update optimizer learning rates
                for g in optimizer.param_groups:
                    g['lr'] = learning_rate

            # Compute and apply the perturbations to the image
            noise0, noise1, image0_output,image1_output = model.forward(image0, image1)

            # if mask_constraint == 'within_mask':
            #     noise_output *= class_mask
            # elif mask_constraint == 'out_of_mask':
            #     noise_output *= (1.0 - class_mask)
            #
            # if mask_constraint != 'none':
            #     image_output = torch.clamp(image + noise_output, 0.0, 1.0)
            stereo_output = stereo_model.forward(image0_output, image1_output, logits = logits)

            loss = model.compute_loss(
                depth_output=stereo_output,
                depth_target=ground_truth,
                logits = logits)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if step % 50 == 0:
                print('Loss = {:.5f}'.format(loss.item()), end=' |',flush=True)
            # sys.stdout.write(
            #     'Sample={:3}/{:3}  Step={:3}/{:3}  Loss={:.5f}\r'.format(
            #         idx + 1, n_sample, step, n_step, loss.item()))
            # sys.stdout.flush()
            # params = {"learning_rate": learning_rate}
            # run["parameters"] = params
            # run["train/loss"].log(loss.item())
        return noise0, noise1
