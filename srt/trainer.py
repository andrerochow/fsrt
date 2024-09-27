import torch
import numpy as np
import torch.nn as nn
import os
import math
import srt.utils.visualize as vis

from tqdm import tqdm
from srt.utils.common import mse2psnr, concatenate_dict, gather_all, get_rank, get_world_size
from modules.util import ImagePyramide
from modules.perceptual_loss import *
from collections import defaultdict
    


class FSRTTrainer:
    def __init__(self, model, optimizer, cfg, device, out_dir, kp_detector, discriminator, optimizer_disc):
        self.model = model
        self.kp_detector = kp_detector
        self.optimizer = optimizer
        self.config = cfg
        self.device = device
        self.out_dir = out_dir
        self.phase2_start = cfg['training']['iters_in_phase1']
        self.phase3_start = cfg['training']['iters_in_phase1']+cfg['training']['iters_in_phase2']
        self.disc_warmup_iters = cfg['training']['disc_warmup_iters']
        self.statistical_regularization = cfg['training']['statistical_regularization']
        self.simulate_out_of_frame_motion = cfg['data']['simulate_out_of_frame_motion']
        self.scales = cfg['training']['scales']
        self.pyramid = ImagePyramide(self.scales, 3)
        self.vgg19 = init_perceptual_loss()
        self.perceptual_loss_weight = cfg['training']['perceptual_loss_weight']
        self.lambda_var = cfg['training']['variance_loss_weight']
        self.lambda_cov = cfg['training']['covariance_loss_weight']
        self.lambda_invar= cfg['training']['invariance_loss_weight']
        self.lambda_mse = cfg['training']['mse_loss_weight']
        self.generator_gan_weight = cfg['training']['generator_gan_loss_weight']
        self.discriminator_gan_weight = cfg['training']['discriminator_gan_loss_weight']
        self.generator_gan_feature_matching_weight = cfg['training']['generator_gan_feature_matching']
        self.disc_scales = cfg['discriminator']['scales']
        self.use_disc = cfg['discriminator']['use_disc']
        self.optimizer_disc = optimizer_disc
        self.discriminator = discriminator

    def evaluate(self, val_loader):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        self.model.eval()
        eval_lists = defaultdict(list)

        loader = val_loader if get_rank() > 0 else tqdm(val_loader)
        sceneids = []

        for data in loader:
            sceneids.append(data['sceneid'])
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_lists[k].append(v)

        sceneids = torch.cat(sceneids, 0).cuda()
        sceneids = torch.cat(gather_all(sceneids), 0)

        print(f'Evaluated {len(torch.unique(sceneids))} unique scenes.')
        eval_dict = {k: torch.cat([v_.unsqueeze(0) if v_.dim() == 0 else v_ for v_ in v],dim=0) for k, v in eval_lists.items()} 
        eval_dict = concatenate_dict(eval_dict)  # Concatenate across processes
        eval_dict = {k: v.mean().item() for k, v in eval_dict.items()}  # Average across batch_size
        print('Evaluation results:')
        print(eval_dict)
        return eval_dict

    def train_step(self, data, it):
        self.model.train()
        self.optimizer.zero_grad()
        loss, loss_terms, disc_items = self.compute_loss(data, it)
        loss = loss.mean(0)
        if self.statistical_regularization:
            loss+=loss_terms['reg_loss'].mean()
        loss_terms = {k: v.mean(0).item() for k, v in loss_terms.items()}
        loss.backward()
        self.optimizer.step()
        
        if it >= self.phase3_start and self.use_disc:
            self.optimizer_disc.zero_grad()
            loss_disc = 0
            
            discriminator_maps_generated = self.discriminator(disc_items['generated'], kp=disc_items['target_kp_disc'], detach=True)
            discriminator_maps_real = self.discriminator(disc_items['real'], kp=disc_items['target_kp_disc'])

            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = (1. - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
                loss_disc += self.discriminator_gan_weight * value.mean()
            loss_terms['disc_gan'] = loss_disc
            loss_disc.backward()
            self.optimizer_disc.step()
        return loss.item(), loss_terms


        
    def extract_keypoints_and_expression(self, img_src, img_driv, img_src_augm=None, img_driv_augm=None): 
        '''
        Shapes:
            img_src:       [bs,nsrc,3,h,w]
            img_driv:      [bs,(ndriv),3,h,w]
        '''
        assert self.kp_detector is not None
        if len(img_driv.shape) == 4:
            img_driv = img_driv.unsqueeze(1) 
            if img_driv_augm is not None:
                img_driv_augm = img_driv_augm.unsqueeze(1) 
            
        bs, nsrc, c, h, w = img_src.shape
        nkp = self.kp_detector.num_kp
        ndriv = img_driv.shape[1]    
        img = torch.cat([img_src,img_driv], dim = 1).view(-1,c,h,w)
        if img_src_augm is not None:
            img_augm = torch.cat([img_src_augm,img_driv_augm], dim = 1).view(-1,c,h,w)

        with torch.no_grad():
            kps, latent_dict = self.kp_detector(img)
            kps = kps.view(bs,nsrc+ndriv,nkp,2)
            heatmaps = latent_dict['heatmap'].view(bs,nsrc+ndriv,nkp,latent_dict['heatmap'].shape[-2],latent_dict['heatmap'].shape[-1])
            feature_maps = latent_dict['feature_map'].view(bs,nsrc+ndriv,latent_dict['feature_map'].shape[-3],latent_dict['feature_map'].shape[-2],latent_dict['feature_map'].shape[-1])
            
        if img_src_augm is not None:
            with torch.no_grad():
                _, latent_dict_augm = self.kp_detector(img_augm)
                heatmaps_augm = latent_dict_augm['heatmap'].view(bs,nsrc+ndriv,nkp,latent_dict_augm['heatmap'].shape[-2],latent_dict_augm['heatmap'].shape[-1])
                feature_maps_augm = latent_dict_augm['feature_map'].view(bs,nsrc+ndriv,latent_dict_augm['feature_map'].shape[-3],latent_dict_augm['feature_map'].shape[-2],latent_dict_augm['feature_map'].shape[-1])
            
            
        kps_src, kps_driv = torch.split(kps,[nsrc,ndriv], dim=1)
        _, heatmap_driv = torch.split(heatmaps,[nsrc,ndriv], dim=1)
        _, feature_map_driv = torch.split(feature_maps,[nsrc,ndriv], dim=1)
        
        if kps_driv.shape[1] == 1:
            kps_driv = kps_driv.squeeze(1)
        
        expression_vector_src , expression_vector = torch.split(self.model.expression_encoder(feature_maps.flatten(0,1),heatmaps.flatten(0,1)).view(bs,nsrc+ndriv,-1), [nsrc,ndriv], dim = 1) 
        if expression_vector.shape[1] == 1:
            expression_vector = expression_vector.squeeze(1)
            
        if img_src_augm is not None:
            expression_vector_src_augm , expression_vector_augm = torch.split(self.model.expression_encoder(feature_maps_augm.flatten(0,1),heatmaps_augm.flatten(0,1)).view(bs,nsrc+ndriv,-1), [nsrc,ndriv], dim = 1) 
            if expression_vector_augm.shape[1] == 1:
                expression_vector_augm = expression_vector_augm.squeeze(1)
        else:
            expression_vector_src_augm = expression_vector_augm = None

        return kps_src, kps_driv, expression_vector, expression_vector_src, expression_vector_augm, expression_vector_src_augm
        

    def compute_loss(self, data, it):
        device = self.device

        input_images_augm = data.get('input_images_augm').to(device)
        input_images_augm2 = data.get('input_images_augm2').to(device)
        input_pos = data.get('input_pos').to(device)
        target_image_augm = data.get('target_image_augm').to(device)
        target_image_augm2 = data.get('target_image_augm2').to(device)
        target_pixels = data.get('target_pixels_augm').to(device)
        target_pos = data.get('target_pos').to(device)
        remaining_pos = data.get('remaining_pos').to(device)
        
        #Randomly select one augemented version for expression extraction
        if torch.rand(1) < 0.5:
            target_image_rand = target_image_augm2
        else:
            target_image_rand = target_image_augm
        #Also select the cropped driving frame (see Paper Sec. 3.2 in ¶ Cropping).
        target_image_augm3_crop = data.get('target_image_augm3_crop').to(device)
        input_kps, target_kps, expression_vector, expression_vector_src_augm2, expression_vector_augm3_crop, expression_vector_src_augm = self.extract_keypoints_and_expression(input_images_augm2, target_image_rand, input_images_augm, target_image_augm3_crop)

        del input_images_augm2
        
        if self.simulate_out_of_frame_motion:
            #Estimate the keypoints coordinates of the uncropped images
            input_kps, target_kps, _, _, _, _ = self.extract_keypoints_and_expression(data.get('input_images_augm2_uncropped').to(device),data.get('target_image_augm2_uncropped').to(device))
            #Transform the keypoints into the coordinate system of the cropped images
            kp_scale = data.get('kp_scale').to(device)
            kp_shift = data.get('kp_shift').to(device)
            input_kps = (input_kps+kp_shift[:,None,None])*kp_scale[:,None,None]
            target_kps = (target_kps+kp_shift[:,None])*kp_scale[:,None]

            
        del target_image_augm
        del target_image_augm2
        del target_image_augm3_crop

        loss_terms = dict()
        
        #Expression vector regularization
        d = expression_vector.shape[-1]
        expression_vector_gathered_1 = torch.cat([expression_vector_src_augm,expression_vector.unsqueeze(1)], dim=1).view(-1,d) 
        expression_vector_gathered_2 = torch.cat([expression_vector_src_augm2,expression_vector_augm3_crop.unsqueeze(1)], dim=1).view(-1,d) 

        if self.statistical_regularization:
            #1. Variance along feature dimension
            weight = expression_vector_gathered_1.shape[0]
            S_1 = torch.sqrt(torch.var(expression_vector_gathered_1,dim=-1)+0.0001)
            S_2 = torch.sqrt(torch.var(expression_vector_gathered_2,dim=-1)+0.0001)
            v_1 = (1./weight)*torch.nn.functional.relu(1.-S_1).sum()
            v_2 = (1./weight)*torch.nn.functional.relu(1.-S_2).sum()

            #2.Covariance and variance (diagonal) 
            cov_sq_1 = torch.cov(expression_vector_gathered_1.T)**2
            cov_sq_2 = torch.cov(expression_vector_gathered_2.T)**2
            c_1 = (1./d)*cov_sq_1.sum()
            c_2 = (1./d)*cov_sq_2.sum()
        else:
            c_1 = 0
            c_2 = 0
            v_1 = 0
            v_2 = 0 
            
        #3.Invariance criterion
        s = 0.5*(((expression_vector_src_augm-expression_vector_src_augm2)**2).mean() + ((expression_vector-expression_vector_augm3_crop)**2).mean())
        reg_loss = self.lambda_invar*s + self.lambda_var*(v_1+v_2) + self.lambda_cov*(c_1+c_2)
        loss_terms['reg_loss'] = reg_loss

        
        #With a probability of 75% we select the expression vector of target_image_augm3_crop
        selected = torch.rand(expression_vector.shape[0]) 
        selected = (selected < 0.25) 
        expression_vector_selected = expression_vector.clone()
        expression_vector_selected[selected.bool()] = expression_vector[selected.bool()]
        expression_vector_selected[(1-selected.type('torch.LongTensor')).bool()] = expression_vector_augm3_crop[(1-selected.type('torch.LongTensor')).bool()]
        
        #Encode input_images_augm along with the expression vector of the same source image with the different color augmentation (expression_vector_src_augm2)
        z = self.model.encoder(input_images_augm, input_kps, input_pos, expression_vector=expression_vector_src_augm2)

        bs, nsrc, c, h, w = input_images_augm.shape
        if data.get('remaining_idxs').shape[1] > 0:
            with torch.no_grad():
                pred_pixels_remaining, extras_remaining = self.model.decoder(z.detach(), remaining_pos, target_kps[:,None].repeat(1,remaining_pos.shape[1],1,1), expression_vector=expression_vector_selected)
                del remaining_pos
            pred_pixels_, extras = self.model.decoder(z, target_pos, target_kps[:,None].repeat(1,target_pos.shape[1],1,1), expression_vector=expression_vector_selected)

            all_idxs = torch.cat([data.get('sampled_idxs').to(device),data.get('remaining_idxs').to(device)], dim = -1)
            all_preds = torch.cat([pred_pixels_,pred_pixels_remaining], dim = -2)
            pred_pixels = torch.zeros_like(all_preds, device=device)
        
            for i in range(bs):
                pred_pixels[i][all_idxs[i]] = all_preds[i]
        else:
            pred_pixels, extras = self.model.decoder(z, target_pos, target_kps[:,None].repeat(1,target_pos.shape[1],1,1), expression_vector=expression_vector_selected)
        
        #Loss functions
        loss = 0.
        loss = loss + self.lambda_mse*((pred_pixels - target_pixels)**2).mean((1, 2))
        loss_terms['mse'] = loss.detach().clone()
        
        generated=None
        real=None
        target_kp_disc=None
        
        if it >= self.phase2_start:
            pred_pixels = pred_pixels.view(bs,h,w,c).permute(0,3,1,2)
            target_pixels = target_pixels.view(bs,h,w,c).permute(0,3,1,2)

            if sum(self.perceptual_loss_weight) > 0.:
                perc_loss = 0
                x_vgg = self.vgg19(pred_pixels)
                y_vgg = self.vgg19(target_pixels)

                for i, weight in enumerate(self.perceptual_loss_weight):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    perc_loss += weight * value

                loss_terms['perceptual'] = perc_loss
                loss+=perc_loss
                
        if it >= self.phase3_start and self.use_disc:
            generated = self.pyramid(pred_pixels)
            real = self.pyramid(target_pixels)
            target_kp_disc = target_kps.detach()
            
            if it >= self.phase3_start + self.disc_warmup_iters and (self.generator_gan_weight != 0 or sum(self.generator_gan_feature_matching_weight) != 0):
                discriminator_maps_generated = self.discriminator(generated, kp=target_kp_disc)
                discriminator_maps_real = self.discriminator(real, kp=target_kp_disc)
                if self.generator_gan_weight != 0:
                    value_total = 0
                    for scale in self.disc_scales:
                        key = 'prediction_map_%s' % scale
                        value = ((1. - discriminator_maps_generated[key]) ** 2).mean()
                        value_total += self.generator_gan_weight * value
                        loss+=value_total
                        loss_terms['gen_gan'] = value_total.detach().clone()

                if sum(self.generator_gan_feature_matching_weight) != 0:
                    value_total = 0
                    for scale in self.disc_scales:
                        key = 'feature_maps_%s' % scale
                        for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                            if self.generator_gan_feature_matching_weight[i] == 0:
                                continue
                            value = torch.abs(a - b).mean()
                            value_total += self.generator_gan_feature_matching_weight[i] * value
                        loss+=value_total
                        loss_terms['feature_matching'] = value_total.detach().clone()            

        return loss, loss_terms, {'generated': generated, 'real': real, 'target_kp_disc': target_kp_disc }
    
    def eval_step(self, data):
        self.model.eval()
        with torch.no_grad():
            loss, loss_terms, disc_items = self.compute_loss(data, self.phase2_start)

        mse = loss_terms['mse']
        psnr = mse2psnr(mse)
        return {'psnr': psnr, 'mse': mse, **loss_terms}


    def render_face(self, z, target_kps, target_pos, expression_vector=None):
        batch_size, height, width = target_pos.shape[:3]
        target_pos = target_pos.flatten(1, 2)
        target_kps = target_kps.unsqueeze(1).repeat(1, target_pos.shape[1], 1,1)

        max_num_rays = self.config['data']['num_pixels'] * \
                self.config['training']['batch_size'] // (target_pos.shape[0] * get_world_size())
        num_rays = target_pos.shape[1]
        img = torch.zeros((target_pos.shape[0],target_pos.shape[1],3))
        all_extras = []
        for i in range(0, num_rays, max_num_rays):
            img[:, i:i+max_num_rays], extras = self.model.decoder(
                z, target_pos[:, i:i+max_num_rays], target_kps[:, i:i+max_num_rays], expression_vector=expression_vector,
            )
         
        img = img.view(img.shape[0], height, width, 3)
        return img


    def visualize_face(self, data, mode='val'):
        self.model.eval()

        with torch.no_grad():
            device = self.device
            input_images_augm = data.get('input_images_augm').to(device)
            input_pos = data.get('input_pos').to(device)
            target_pos = input_pos[:,0].clone().to(device)
            target_image_augm = data.get('target_image_augm').to(device)
            
            input_kps, target_kps, expression_vector_augm, expression_vector_src_augm,_,_ = self.extract_keypoints_and_expression(input_images_augm, target_image_augm)
            
            input_images_np = np.transpose(input_images_augm.cpu().numpy(), (0, 1, 3, 4, 2))

            z = self.model.encoder(input_images_augm, input_kps, input_pos, expression_vector=expression_vector_src_augm)

            batch_size, num_input_images, height, width, _ = input_pos.shape

            columns = []
            for i in range(num_input_images):
                header = 'input' if num_input_images == 1 else f'input {i+1}'
                columns.append((header, input_images_np[:, i], 'image'))
                
            img = self.render_face(z, target_kps, target_pos, expression_vector=expression_vector_augm)
            name = 'driving'
            columns.append((f'render {name}', img.cpu().numpy(), 'image'))
            t_im = target_image_augm.cpu().numpy().transpose(0,2,3,1)
            columns.append((f'GT {name}', t_im, 'image'))

            output_img_path = os.path.join(self.out_dir, f'renders-{mode}')
            vis.draw_visualization_grid(columns, output_img_path)
