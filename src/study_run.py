import os

import sys
import global_constants as settings
from perturb_main import run
from utils import pro_dir


#TODO: remove 때는 iteration 늘림
#TODO: 원래 MONO에서 target attak에서 제한 걸던 부분 이해 및 복구


n_steps = 50
learning_rates =[4.0, 2.0, 1.0]
learning_schedule = [550, 650, 100000]
transform_method = 'remove'
image0_path = os.path.join(pro_dir, 'testing/kitti_scene_flow_test_image0_1.txt')
image1_path = os.path.join(pro_dir, 'testing/kitti_scene_flow_test_image1_1.txt')
ground_truth_path = os.path.join(pro_dir, 'testing/kitti_scene_flow_test_objects_1.txt')
# ground_truth_path = os.path.join(pro_dir, 'testing/kitti_scene_flow_test_disparity_10.txt')
n_height = 256
n_width = 640
output_norm = 16/255
perturb_method = 'target'
perturb_mode = 'both'
stereo_method = 'psmnet'

output_path = os.path.join(pro_dir, 'perturb_models/{}/target/{}_test'.format(stereo_method,transform_method))
device = 'gpu'





if stereo_method == 'psmnet':
        stereo_model_restore_path = os.path.join(pro_dir, 'pretrained_models/PSMNet/pretrained_model_KITTI2015.tar')
elif stereo_method == 'deeppruner':
        stereo_model_restore_path = os.path.join(pro_dir, 'pretrained_models/DeepPruner/DeepPruner-best-kitti.tar')
elif stereo_method == 'aanet':
        stereo_model_restore_path = os.path.join(pro_dir, 'pretrained_models/AANet/aanet_kitti15-fb2a0d23.pth')

if __name__ == '__main__':
        run(image0_path = image0_path,
                image1_path = image1_path,
                ground_truth_path= ground_truth_path,
                # Run settings
                n_height=n_height,
                n_width=n_width,
                n_step=n_steps,
                learning_rate=learning_rates,
                learning_schedule=learning_schedule,
                output_norm=output_norm,
                # Perturb method settings
                perturb_method=perturb_method,
                transform_method = transform_method,
                perturb_mode=perturb_mode,
                # Stereo model settings
                stereo_method=stereo_method,
                stereo_model_restore_path=stereo_model_restore_path,
                # Output settings
                output_path=output_path,
                # Hardware settings
                device='gpu')