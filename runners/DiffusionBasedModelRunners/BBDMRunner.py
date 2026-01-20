import os

import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn

from PIL import Image
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel_c2v import BrownianBridgeModel_c2v

from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid
from tqdm.autonotebook import tqdm
from torchsummary import summary
import random
import matplotlib.pyplot as plt
import numpy as np
import torch


@Registers.runners.register_with_name('c2v_BBDMRunner')
class c2v_BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM_c2v":
            bbdmnet = BrownianBridgeModel_c2v(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = super().load_model_from_checkpoint()


    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               # verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler))
        return [optimizer], [scheduler]
    

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        return model_states, optimizer_scheduler_states

    
    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        (x, x_name), (x_cond, x_cond_name), *_ = batch
        x = x.to(self.config.training.device[0])
        x_cond = x_cond.to(self.config.training.device[0])

        if len(batch) > 3:
            (condition, condition_name) = batch[2]
            condition = condition.to(self.config.training.device[0])
        else:
            condition = None

        loss, additional_info = net(x, x_cond, condition=condition)
        
        if write:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            if additional_info.__contains__('recloss_noise'):
                self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
            if additional_info.__contains__('recloss_xy'):
                self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
        return loss
    
        
    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        reverse_sample_path = make_dir(os.path.join(sample_path, 'reverse_sample'))
        reverse_one_step_path = make_dir(os.path.join(sample_path, 'reverse_one_step_samples'))

        (x, x_name), (x_cond, x_cond_name), *_ = batch

        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond = x_cond[0:batch_size].to(self.config.training.device[0])

        if len(batch) > 3:
            (condition, condition_name) = batch[2]
            condition = condition[0:batch_size].to(self.config.training.device[0])
        else:
            condition = None

        grid_size = 4

        # Debug: Check if model parameters have changed from initialization
        if stage != 'test' and self.global_step % 1000 == 0:
            param_sum = sum(p.abs().sum().item() for p in net.parameters())
            print(f"[Debug] Model parameter sum at step {self.global_step}: {param_sum:.2f}")

        # Generate sample
        sample = net.sample(x_cond, condition=condition, clip_denoised=self.config.testing.clip_denoised).to('cpu')
        
        # FIXED: Use correct dimension indexing [batch, channel, depth, height, width]
        depth_slice = sample.shape[2] // 2  # Middle slice along depth dimension
        
        # Move all data to CPU for visualization
        x_cpu = x.to('cpu')
        x_cond_cpu = x_cond.to('cpu')
        sample_cpu = sample
        
        # Extract middle slices
        sample_slice = sample_cpu[:, :, depth_slice, :, :]
        cond_slice = x_cond_cpu[:, :, depth_slice, :, :]
        gt_slice = x_cpu[:, :, depth_slice, :, :]
        
        # Create individual image grids
        to_normal = self.config.data.dataset_config.to_normal
        sample_grid = get_image_grid(sample_slice, grid_size, to_normal=to_normal)
        cond_grid = get_image_grid(cond_slice, grid_size, to_normal=to_normal)
        gt_grid = get_image_grid(gt_slice, grid_size, to_normal=to_normal)
        
        # Save individual images (backward compatibility)
        Image.fromarray(sample_grid).save(os.path.join(sample_path, 'generated_sample.png'))
        Image.fromarray(cond_grid).save(os.path.join(sample_path, 'input_condition.png'))
        Image.fromarray(gt_grid).save(os.path.join(sample_path, 'ground_truth.png'))
        
        # Compute metrics to verify the model is actually generating different outputs
        mse_cond_sample = torch.nn.functional.mse_loss(sample_slice, cond_slice).item()
        mse_sample_gt = torch.nn.functional.mse_loss(sample_slice, gt_slice).item()
        mse_cond_gt = torch.nn.functional.mse_loss(cond_slice, gt_slice).item()
        
        # Create side-by-side comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        title_text = f'{stage.capitalize()} - Step {self.global_step} | MSE(Cond→Gen)={mse_cond_sample:.4f}, MSE(Gen→GT)={mse_sample_gt:.4f}, MSE(Cond→GT)={mse_cond_gt:.4f}'
        fig.suptitle(title_text, fontsize=14, fontweight='bold')
        
        axes[0].imshow(cond_grid)
        axes[0].set_title('Input Condition (ULF)', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(sample_grid)
        axes[1].set_title('Generated Output', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(gt_grid)
        axes[2].set_title('Ground Truth (HF)', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(sample_path, f'{stage}_comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log metrics to tensorboard
        if stage != 'test':
            self.writer.add_scalar(f'{stage}_metrics/mse_condition_to_generated', mse_cond_sample, self.global_step)
            self.writer.add_scalar(f'{stage}_metrics/mse_generated_to_gt', mse_sample_gt, self.global_step)
            self.writer.add_scalar(f'{stage}_metrics/mse_condition_to_gt', mse_cond_gt, self.global_step)
            
            # Warning if generated output is too similar to input condition
            if mse_cond_sample < 0.001:
                print(f"⚠️ WARNING: Generated output is nearly identical to input condition! MSE={mse_cond_sample:.6f}")
                print(f"   This suggests the model may not be learning or generating properly.")
                print(f"   Check: 1) Model training progress, 2) Loss values, 3) Sampling parameters")
        
        # Log to tensorboard
        if stage != 'test':
            self.writer.add_image(f'{stage}_generated', sample_grid, self.global_step, dataformats='HWC')
            self.writer.add_image(f'{stage}_condition', cond_grid, self.global_step, dataformats='HWC')
            self.writer.add_image(f'{stage}_ground_truth', gt_grid, self.global_step, dataformats='HWC')
            
            # Also log the comparison figure
            comparison_img = plt.imread(comparison_path)
            if comparison_img.dtype == np.float32 or comparison_img.dtype == np.float64:
                comparison_img = (comparison_img * 255).astype(np.uint8)
            self.writer.add_image(f'{stage}_comparison', comparison_img, self.global_step, dataformats='HWC')

    
    @torch.no_grad()
    def sample_results_c2v(self, net, test_batch):
        (x, x_name) = test_batch[0]
        (x_cond, x_cond_name) = test_batch[1]
        x = x.to(self.config.training.device[0])
        x_cond = x_cond.to(self.config.training.device[0])

        if len(test_batch) > 2:
            (condition, condition_name) = test_batch[2]
            condition = condition.to(self.config.training.device[0])
        else:
            condition = None

        sample = net.sample(x_cond, condition=condition, clip_denoised=self.config.testing.clip_denoised)
        
        return x, sample
        
    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):
        pass


