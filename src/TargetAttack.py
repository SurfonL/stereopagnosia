import torch
import global_constants as settings
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable

EPSILON = 1e-8


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1)
        print(self.disp)
        return out

class TargetPerturbations(torch.nn.Module):
    '''
    Base perturbations image class

    Args:
        n_height : int
            height of perturbations image
        n_width : int
            width of perturbations image
        n_channel : int
            number of channels of perturbations image
        output_norm : float
            upper norm of perturbations
    '''

    def __init__(self,
                 n_height=settings.N_HEIGHT,
                 n_width=settings.N_WIDTH,
                 n_channel=settings.N_CHANNEL,
                 output_norm=settings.OUTPUT_NORM):
        super(TargetPerturbations, self).__init__()

        self.output_norm = output_norm

        # Initialize noise image
        self.weights0 = torch.nn.Parameter(
            torch.zeros([n_channel, n_height, n_width],
            requires_grad=True))
        self.weights1 = torch.nn.Parameter(
            torch.zeros([n_channel, n_height, n_width],
                        requires_grad=True))

    def forward(self, image0, image1):
        # Clip the perturbations outside of upper norm
        noise0 = torch.clamp(self.weights0, -self.output_norm, self.output_norm)
        noise1 = torch.clamp(self.weights1, -self.output_norm, self.output_norm)

        image0_output = image0 + noise0
        image1_output = image1 + noise1
        return noise0, noise1, image0_output, image1_output


class TargetPerturbationsModel(object):
    '''
    Perturbations model class

    Args:
        n_height : int
            height of perturbations image
        n_width : int
            width of perturbations image
        n_channel : int
            number of channels of perturbations image
        output_norm : float
            upper norm of perturbations
        device : torch.device
            device to run on
    '''

    def __init__(self,
                 n_height=settings.N_HEIGHT,
                 n_width=settings.N_WIDTH,
                 n_channel=settings.N_CHANNEL,
                 output_norm=settings.OUTPUT_NORM,
                 device=torch.device(settings.CUDA)):

        self.device = device

        self.perturbations = TargetPerturbations(
            n_height=n_height,
            n_width=n_width,
            n_channel=n_channel,
            output_norm=output_norm)

        # Move to device
        self.to(self.device)


        self.dispairtyregress = disparityregression(192)

    def forward(self, image0, image1):
        '''
        Applies perturbations to image

        Args:
            image : tensor
                N x C x H x W tensor

        Returns:
            tensor : N x C x H x W perturbations
            tensor : N x C x H x W perturbed image
        '''

        # Apply perturbations to image
        # TODO: 1
        noise0, noise1, image0_pert, image1_pert = self.perturbations(image0,image1)

        # Clip image between 0.0 and 1.0 and recover the noise
        # TODO: 2
        image0_pert = torch.clamp(image0_pert, 0.0, 1.0)
        image1_pert = torch.clamp(image1_pert, 0.0, 1.0)

        return noise0, noise1, image0_pert, image1_pert

    def compute_loss(self, depth_output, depth_target, logits = False):
        '''
        Computes target consistency loss

        Args:
            depth_output : tensor
                N x 1 x H x W source tensor
                N x D x H x W if logits = True
            depth_target : tensor
                N x 1 x H x W target tensor

        Returns:
            float : loss
        '''

        if logits:
            depth_output2 = depth_output
            depth_target2 = depth_target.round().long()


            a1 = depth_output2[0, :, 150, 200]
            a2 = F.softmax(a1)
            d = torch.from_numpy(np.array(range(192))).cuda().float()
            print(a1[9:25])
            print(torch.matmul(a2, d))
            print(a2[20:24])

            # b1 = depth_target2[0, :, 150, 200]
            # b2 = F.softmax(b1)
            # print(torch.matmul(b2, d))





            g = depth_output2.gather(1,depth_target2)

            loss = g.sum()*(-1/(256*640))

            return loss
        else:
            # Compute target depth consistency loss function
            delta = torch.abs(depth_target - depth_output)
            delta = delta / (depth_target + EPSILON)
            loss = torch.mean(delta)

        return loss

    def to(self, device):
        '''
        Moves model to device

        Args:
            device : torch.device
        '''

        self.perturbations.to(device)

    def train(self):
        '''
        Sets model to training mode
        '''

        self.perturbations.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.perturbations.eval()

    def parameters(self):
        '''
        Fetches model parameters

        Returns:
            list : list of model parameters
        '''

        return list(self.perturbations.parameters())

class Transformation:

    @classmethod
    def flip_horizontal(cls,img):
        "img in tensor"
        return torch.flip(img, dims=[-1])

    @classmethod
    def multiply_object(cls, depth, obj_mask, scale = 2):
        depth = depth.detach().cpu().numpy()

        obj_mask = torch.squeeze(obj_mask).detach().cpu().numpy()
        obj_mask = Image.fromarray(obj_mask).resize((depth.shape[3], depth.shape[2]), resample=Image.NONE)
        obj_mask = np.array(obj_mask).astype(np.float32)
        obj_mask = cv2.medianBlur(obj_mask, 3)

        idx = obj_mask == np.min(obj_mask[np.nonzero(obj_mask)])
        obj_mask[idx] = scale
        obj_mask[np.logical_not(idx)] = 1

        target = np.multiply(depth, obj_mask)

        target = torch.from_numpy(target).cuda()

        return target

    @classmethod
    def remove_object(cls, depth, obj_mask, window = 10):
        depth = depth.detach().cpu().numpy()
        obj_mask = torch.squeeze(obj_mask).detach().cpu().numpy()
        obj_mask = Image.fromarray(obj_mask).resize((depth.shape[3], depth.shape[2]), resample=Image.NONE)
        obj_mask = np.array(obj_mask)
        obj_mask = cv2.medianBlur(obj_mask, 3)

        obj_no = np.min(obj_mask[np.nonzero(obj_mask)])
        _,_,h,w = depth.shape
        #bolder object so it covers the whole depth
        mask_resize = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                if obj_mask[i, j] == obj_no:
                    mask_resize[i, j] = obj_no
                    win = min(window,i,j,h-1-i,w-1-j)
                    mask_resize[i - win:i + win+1, j - win:j + win+1] = obj_no

        #reverse pixels, as ㅑㅅ will be multipied to depth
        idx = mask_resize == obj_no
        mask_resize[idx] = 0
        mask_resize[np.logical_not(idx)] = 1

        #apply mask
        removed = np.multiply(depth, mask_resize)



        #fill in the removed object
        #searches from the left where the object boundary is
        restored = removed.copy()
        bound_l = []
        bound_r = []
        for i in range(h):
            for j in range(1, w - 1):
                if (restored[0,0,i, j] == 0 and restored[0,0,i, j - 1] != 0) or (restored[0,0,i, 0] == 0 and j==1):
                    bound_l.append((i, j - 1))
                if (restored[0,0,i, j] == 0 and restored[0,0,i, j + 1] != 0) or (restored[0,0,i, w-1] == 0 and j == w-2):
                    bound_r.append((i, j + 1))

        #fill in the blank with the minimum of the far-left and far-right. b.c object may overlap
        for c1, c2 in zip(bound_l, bound_r):
            left = restored[0, 0, c1[0], c1[1]]
            right = restored[0, 0, c2[0], c2[1]]

            left = 10000 if left == 0 else left
            right = 10000 if right == 0 else right

            min_dep = np.minimum(left, right)
            restored[0, 0, c1[0]:c2[0] + 1, c1[1]:c2[1] + 1] = min_dep

        restored = torch.from_numpy(restored.astype(np.float32)).cuda()
        return restored

    @classmethod
    def create_object(cls, depth):
        depth = depth.detach().cpu().numpy()
        import imageio
        j=136
        obj_mask = imageio.imread(
            'C:\\Users\\Woo\\Desktop\\Research_Projects\\stereopagnosia\\data\\kitti_scene_flow\\training\\obj_map\\000{}_10.png'.format(
                j))
        create_disp = np.load('C:\\Users\\Woo\\Desktop\\Research_Projects\\stereopagnosia\\perturb_models/{}/target/{}/disparity_origin/00{}.npy'.format('psmnet', 'multiply_paper', j))
        sp = (640, 256)
        obj_mask = Image.fromarray(obj_mask).resize(sp, resample=Image.NONE)
        obj_mask = np.array(obj_mask)
        obj_mask = cv2.medianBlur(obj_mask, 9)

        x, y = np.where(obj_mask == 2)

        new_disp = depth.copy()
        new_disp[0,0, x, y + 150] = create_disp[x, y]
        new_disp = torch.from_numpy(new_disp.astype(np.float32)).cuda()

        return new_disp