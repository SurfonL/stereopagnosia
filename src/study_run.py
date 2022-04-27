import global_constants as settings
from perturb_main import run
from utils import pro_dir
import os


image0_path = os.path.join(pro_dir, 'testing/kitti_scene_flow_test_image0.txt')
image1_path = os.path.join(pro_dir, 'testing/kitti_scene_flow_test_image1.txt')
ground_truth_path = os.path.join(pro_dir, 'testing/kitti_scene_flow_test_disparity.txt')
n_height = 256
n_width = 640
output_norm = 0.02
perturb_method = 'fgsm'
perturb_mode = 'both'
stereo_method = 'psmnet'
stereo_model_restore_path = os.path.join(pro_dir, 'pretrained_models/PSMNet/pretrained_model_KITTI2015.tar')
output_path = os.path.join(pro_dir, 'perturb_models/psmnet/fgsm/both_norm2e2')
device = 'gpu'

if __name__ == '__main__':
        run(image0_path = image0_path,
                image1_path = image1_path,
                ground_truth_path= ground_truth_path,
                # Run settings
                n_height=n_height,
                n_width=n_width,
                # Perturb method settings
                perturb_method=perturb_method,
                perturb_mode=perturb_mode,
                # Stereo model settings
                stereo_method=stereo_method,
                stereo_model_restore_path=stereo_model_restore_path,
                # Output settings
                output_path=output_path,
                # Hardware settings
                device='gpu')