import numpy as np
import torch
import os
import glob

from torch.utils.data import Dataset
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from imageio import mimread
from srt.data.augmentation import AllAugmentationTransform, ImageFlipper

    
def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video_loaded = False
        while not video_loaded:
            try:             
                video = np.array(mimread(name,memtest=False))
                video_loaded = True                       
            except:                                          
                video_loaded = False
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array



class VoxDataset(Dataset):

    def __init__(self, 
                 path,
                 kp_path,
                 mode,
                 num_src=1,
                 num_pixels=16384,
                 num_pixels_phase1=4096,
                 frame_shape=(256, 256, 3),
                 id_sampling=True, 
                 is_train=True,
                 augmentation_params=None,
                 image_subsampling=True,
                 simulate_out_of_frame_motion=True,
                 phase1=True):
        
        self.num_src = num_src
        self.phase1 = phase1
        self.root_dir = path
        self.kp_path = kp_path
        self.videos = os.listdir(self.root_dir)
        self.frame_shape = tuple(frame_shape)
        self.id_sampling = id_sampling
        self.image_subsampling = image_subsampling
        if mode == 'val':
            self.is_train = False
            self.id_sampling = False
        else:
            self.is_train = True
        self.simulate_out_of_frame_motion = simulate_out_of_frame_motion
        if self.image_subsampling:
            assert frame_shape[0]==frame_shape[1] #Image subsampling is only implemented for quadratic frame shapes
            self.patch_size = np.sqrt(num_pixels)
            assert int(self.patch_size) >= self.patch_size #must be an integer
            self.patch_size = int(self.patch_size)
            assert np.mod(frame_shape[0],self.patch_size) == 0 #frame_shape must be a multiple of the patchsize
        self.num_pixels = num_pixels
        self.num_pixels_phase1 = num_pixels_phase1
        
        _, y, x = np.meshgrid(np.zeros(self.num_src+1), np.arange(frame_shape[0]), np.arange(frame_shape[1]), indexing="ij")
        self.idx_grids = np.stack([x, y], axis=-1).astype(np.float32)
        #Normalize Grid
        self.idx_grids[...,0] = (self.idx_grids[...,0]+0.5 -((frame_shape[0]/2.0)))/(((frame_shape[0]/2.0)))
        self.idx_grids[...,1] = (self.idx_grids[...,1]+0.5 -((frame_shape[1]/2.0)))/(((frame_shape[1]/2.0)))

        
        if id_sampling:
            train_videos = {os.path.basename(video).split('#')[0] for video in
                            os.listdir(os.path.join(self.root_dir, 'train'))}
            train_videos = list(train_videos)
        else:
            train_videos = os.listdir(os.path.join(self.root_dir, 'train'))
        val_videos = os.listdir(os.path.join(self.root_dir, 'val'))
        self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'val')
        
        if self.is_train:
            self.videos = train_videos
        else:
            self.videos = val_videos

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
            self.flipper = ImageFlipper(augmentation_params['flip_param'])
        else:
            self.transform = None
            self.flipper = None
    def __len__(self):
        if self.is_train:
            return len(self.videos)*10000     
        else:
            return len(self.videos)
           
    def __getitem__(self, idx):
        is_null_vec = True
        idx = idx%len(self.videos)
        while is_null_vec: #Identify and exclude broken keypoints
            if self.is_train and self.id_sampling:
                name = self.videos[idx]
                path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
            else:
                name = self.videos[idx]
                path = os.path.join(self.root_dir, name)

            video_name = os.path.basename(path)

            if self.is_train and os.path.isdir(path):
                frames = os.listdir(path)
                num_frames = len(frames)
                frame_idx = np.random.choice(num_frames, replace=True, size=self.num_src+1)
                video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
            else:
                video_array = read_video(path, frame_shape=self.frame_shape)
                num_frames = len(video_array)
                frame_idx = np.random.choice(num_frames, replace=True, size=self.num_src+1)
                video_array_sorted = video_array
            video_array = video_array[frame_idx]
            video_array = np.array(video_array)
            #50% chance for simulating out-of-frame motion.
            #We crop the Image to obtain driving keypoints outside the image (see Sec. 7.2. -> ¶ "Visualizing Out-of-frame Motion" in Supplementary Material). 
            crop_image = (torch.rand(1) < 0.5)
            
            # The follolwing block was removed as it is unnecessary. However, we utilized it for the training of our model ablations in the paper.
            # BEGIN Block
            #if video_array[0].shape[0] != self.frame_shape[0]:
                #if video_array[0].shape[0] < 1.25*self.frame_shape[0] or not crop_image:
                    #video_array = torch.nn.functional.interpolate(torch.from_numpy(video_array.transpose(0,3,1,2)),(self.frame_shape[0],self.frame_shape[1]),mode='bilinear').numpy().transpose(0,2,3,1)
                #else: #Resize the video to twice the output resulution, since we will crop it later
                    #video_array = torch.nn.functional.interpolate(torch.from_numpy(video_array.transpose(0,3,1,2)),(2*self.frame_shape[0],2*self.frame_shape[1]),mode='bilinear').numpy().transpose(0,2,3,1)
                #video_array = [video_array[i] for i in range(video_array.shape[0])]
            # END Block
            
            flipped = False
            if self.flipper is not None:
                video_array, flipped = self.flipper(video_array)
            if self.transform is not None:
                # Create 3 different augmentations
                video_array_augm = self.transform(video_array)
                video_array_augm2 = self.transform(video_array)
                video_array_augm3_crop = self.transform(video_array)

            else:
                #This happens in validation mode
                video_array_augm = video_array
                video_array_augm2 = video_array
                video_array_augm3_crop = video_array
            
            #Try to load the stored keypoints
            kp_path = path.split("/")[-1]
            kp_file = kp_path.split(".")[0] + '.npy'
            try:
                keypoints = np.load(os.path.join(self.kp_path, kp_file))
                keypoints = keypoints[frame_idx] #Shuffle the keypoints in the same way as the video array.
                
                #Identify broken keypoints
                if keypoints.mean(axis=(1, 2)).all() and (type(keypoints) is np.ndarray):
                    is_null_vec = False
                else:
                    idx += 1

                if flipped:
                    #Horizontally flip the keypoints as well
                    keypoints = (keypoints-(255.0/2.0))/(255.0/2.0)
                    keypoints[...,0]*=-1.
                    keypoints=(keypoints+1.)*(255./2.)
            except:
                is_null_vec = True

        out = {}
        
        source_augm = np.array(video_array_augm[:self.num_src], dtype='float32')
        driving_augm = np.array(video_array_augm[self.num_src], dtype='float32')
        source_augm2 = np.array(video_array_augm2[:self.num_src], dtype='float32')
        driving_augm2 = np.array(video_array_augm2[self.num_src], dtype='float32')
        source_augm2_uncropped = source_augm2.copy()
        driving_augm2_uncropped = driving_augm2.copy()
        driving_augm3_crop = np.array(video_array_augm3_crop[self.num_src], dtype='float32')

        # Crop the images to obtain driving keypoints outside the image (see Sec. 7.2. ->  ¶ "Visualizing Out-of-frame Motion" in Supplementary Material).
        kp_scale = np.array([1.,1.], dtype=np.float32)
        kp_shift = np.array([0., 0.],dtype=np.float32)

        if crop_image and self.simulate_out_of_frame_motion: 
            kp_source = keypoints[:self.num_src]
            width_ = kp_source[...,0].max()-kp_source[...,0].min()
            width_stretch = np.rint(0.2*width_)
            height_ = kp_source[...,1].max()-kp_source[...,1].min()
            height_stretch_top = np.rint(0.5*height_)
            height_stretch_bottom = np.rint(0.15*height_)
            #Note that keypoint coordinates are integers and estimated on an image size of 256x256
            min_x = np.rint(max(0,kp_source[...,0].min()-width_stretch))
            max_x = np.rint(min(255,kp_source[...,0].max()+width_stretch))
            min_y = np.rint(max(0,kp_source[...,1].min()-height_stretch_top))
            max_y = np.rint(min(255,kp_source[...,1].max()+height_stretch_bottom))
            
            found_valid_crop = False
            while not found_valid_crop:
                crop_left = torch.randint(0, int(min_x), (1,)).item() if int(min_x) > 0 else 0
                crop_right = torch.randint(0, 256-int(max_x)-1, (1,)).item() if 256-int(max_x)-1> 0 else 0
                crop_top = torch.randint(0, int(min_y), (1,)).item() if int(min_y) > 0 else 0  
                crop_bottom = torch.randint(0, 256-int(max_y)-1, (1,)).item() if 256-int(max_y)-1> 0 else 0
                
                hs,ws,_ = source_augm[0].shape
                crop_top = int(np.rint(crop_top*(hs/256.)))
                crop_bottom = int(np.rint(crop_bottom*(hs/256.)))
                crop_left = int(np.rint(crop_left*(ws/256.)))
                crop_right = int(np.rint(crop_right*(ws/256.)))
                
                #Estimate the keypoint transformation from the uncropped images to the cropped ones. Note keypoints will be normalized to [-1,1].
                kp_shift_in_x = (-crop_left/2.+crop_right/2.)/((ws)/2.)
                kp_shift_in_y = (-crop_top/2.+crop_bottom/2.)/((hs)/2.)  
                kp_scale_in_x = ws/(ws-crop_left-crop_right) #=width/cropped_image_width
                kp_scale_in_y = hs/(hs-crop_top-crop_bottom) #=height/cropped_image_height
                
                # Validate if this is a valid crop, such that the new keypoints can not be outside the range [-2,2]
                if (-1 + kp_shift_in_x)*kp_scale_in_x < -2. or (1 + kp_shift_in_x)*kp_scale_in_x > 2. or (-1 + kp_shift_in_y)*kp_scale_in_y < -2. or (1 + kp_shift_in_y)*kp_scale_in_y > 2.:
                    continue
                else:
                    found_valid_crop = True
                    
            kp_scale = np.array([kp_scale_in_x,kp_scale_in_y], dtype=np.float32)
            kp_shift = np.array([kp_shift_in_x, kp_shift_in_y],dtype=np.float32)
        else:
            crop_left=crop_right=crop_top=crop_bottom=0
            hs,ws,_ = source_augm[0].shape

        if driving_augm.shape[0] != self.frame_shape[0] or driving_augm.shape[1] != self.frame_shape[1] or (crop_left+crop_right+crop_top+crop_bottom) != 0:
            #Finally crop and resize back to self.frame_shape
            source_augm = torch.nn.functional.interpolate(torch.from_numpy(source_augm.transpose(0,3,1,2))[:,:,crop_top:hs-crop_bottom,crop_left:ws-crop_right],(self.frame_shape[0],self.frame_shape[1]),mode='bilinear').numpy().transpose(0,2,3,1)
            source_augm2 = torch.nn.functional.interpolate(torch.from_numpy(source_augm2.transpose(0,3,1,2))[:,:,crop_top:hs-crop_bottom,crop_left:ws-crop_right],(self.frame_shape[0],self.frame_shape[1]),mode='bilinear').numpy().transpose(0,2,3,1)
            source_augm2_uncropped = torch.nn.functional.interpolate(torch.from_numpy(source_augm2_uncropped.transpose(0,3,1,2)),(self.frame_shape[0],self.frame_shape[1]),mode='bilinear').numpy().transpose(0,2,3,1)
            driving_augm = torch.nn.functional.interpolate(torch.from_numpy(driving_augm.transpose(2,0,1)[None])[:,:,crop_top:hs-crop_bottom,crop_left:ws-crop_right],(self.frame_shape[0],self.frame_shape[1]),mode='bilinear')[0].numpy().transpose(1,2,0)
            driving_augm2 = torch.nn.functional.interpolate(torch.from_numpy(driving_augm2.transpose(2,0,1)[None])[:,:,crop_top:hs-crop_bottom,crop_left:ws-crop_right],(self.frame_shape[0],self.frame_shape[1]),mode='bilinear')[0].numpy().transpose(1,2,0)
            driving_augm2_uncropped = torch.nn.functional.interpolate(torch.from_numpy(driving_augm2_uncropped.transpose(2,0,1)[None]),(self.frame_shape[0],self.frame_shape[1]),mode='bilinear')[0].numpy().transpose(1,2,0)


        input_images_augm = source_augm.transpose((0, 3, 1, 2))
        input_images_augm2 = source_augm2.transpose((0, 3, 1, 2))
        input_images_augm2_uncropped = source_augm2_uncropped.transpose((0, 3, 1, 2))


        all_pixels_augm = np.reshape(np.stack(driving_augm, 0), (self.frame_shape[0]  * self.frame_shape[1] , 3))
        all_pos = np.reshape(self.idx_grids[-1], (self.frame_shape[0]  * self.frame_shape[1] , 2))
        num_points = all_pos.shape[0]
        idxs = np.arange(num_points) 

        if self.phase1 and self.is_train:
            replace = num_points < self.num_pixels_phase1
            sampled_idxs = np.random.choice(np.arange(num_points),
                                            size=(self.num_pixels_phase1,),
                                            replace=replace)
            target_pixels_augm = all_pixels_augm[sampled_idxs]
            target_pos = all_pos[sampled_idxs]
            #Create an empty tensor for remaining_idxs and remaining_pos in phase 1
            remaining_idxs = idxs[len(idxs):] 
            remaining_pos = all_pos[all_pos.shape[0]:]

        else:
            if self.image_subsampling:
                idxs = idxs.reshape((self.frame_shape[0],self.frame_shape[1]))
                subsample_factor = self.frame_shape[0]//self.patch_size
                patch_corner_y = np.random.randint(subsample_factor)
                patch_corner_x = np.random.randint(subsample_factor)
                target_pixels_augm = all_pixels_augm
                sampled_idxs = idxs[patch_corner_y::subsample_factor, patch_corner_x::subsample_factor].reshape(self.patch_size**2)
                target_pos = self.idx_grids[-1][patch_corner_y::subsample_factor, patch_corner_x::subsample_factor].reshape(self.patch_size**2,2)
                remaining_idxs = None
                remaining_pos = None
            
                for i in range(subsample_factor):
                    for j in range(subsample_factor):
                        if (i,j) != (patch_corner_x,patch_corner_y):
                            if remaining_pos is None:
                                remaining_pos = self.idx_grids[-1][j::subsample_factor, i::subsample_factor].reshape(self.patch_size**2,2)
                            else:
                                remaining_pos = np.concatenate([remaining_pos,self.idx_grids[-1][j::subsample_factor, i::subsample_factor].reshape(self.patch_size**2,2)], axis = 0)
                            if remaining_idxs is None:
                                remaining_idxs = idxs[j::subsample_factor, i::subsample_factor].reshape(self.patch_size**2)
                            else:
                                remaining_idxs = np.concatenate([remaining_idxs,idxs[j::subsample_factor, i::subsample_factor].reshape(self.patch_size**2)], axis = 0)
                        
        
            else:
                np.random.shuffle(idxs) #Shuffle the indexes
                sampled_idxs = idxs[:self.num_pixels]
                remaining_idxs = idxs[self.num_pixels:]
                target_pixels_augm = all_pixels_augm
                target_pos = all_pos[sampled_idxs]
                remaining_pos = all_pos[remaining_idxs]
                
        driving_augm = driving_augm.transpose((2,0,1))
        driving_augm2 = driving_augm2.transpose((2,0,1))
        driving_augm2_uncropped = driving_augm2_uncropped.transpose((2,0,1))
        driving_augm3_crop = driving_augm3_crop.transpose((2,0,1))


        #Create a driving frame version that is randomly cropped and augmented, which is used for the expression vector extraction (see Paper Sec. 3.2. in ¶ "Cropping").
        scale_x = float(driving_augm3_crop.shape[-1])/self.frame_shape[1]
        scale_y = float(driving_augm3_crop.shape[-2])/self.frame_shape[0]
        
        kp_driv = keypoints[self.num_src]
        width_d = kp_driv[...,0].max()-kp_driv[...,0].min()
        width_stretch_d = 0 
        height_d = kp_driv[...,1].max()-kp_driv[...,1].min()
        height_stretch_top_d = np.rint(0.25*height_d)
        height_stretch_bottom_d = np.rint(0.05*height_d)
        
        t = max(kp_driv[...,1].min()-height_stretch_top_d,0)
        b = min(kp_driv[...,1].max()+height_stretch_bottom_d,self.frame_shape[0]-1)
        l = max(kp_driv[...,0].min()-width_stretch_d,0)
        r = min(kp_driv[...,0].max()+width_stretch_d,self.frame_shape[1]-1)
        
        c_t = torch.randint(0, int(t), (1,)).item() if int(t) > 0 else 0
        c_b = torch.randint(0, self.frame_shape[0]-int(b), (1,)).item() if int(b) < self.frame_shape[0] else 0 
        c_r = torch.randint(0, self.frame_shape[1]-int(r), (1,)).item() if int(r) < self.frame_shape[1] else 0 
        c_l = torch.randint(0, int(l), (1,)).item() if int(l) > 0 else 0
        
        try:
            driving_augm3_crop = torch.nn.functional.interpolate(torch.from_numpy(driving_augm3_crop[None])[:,:,int(np.rint(c_t*scale_y)):int(np.rint(self.frame_shape[0]*scale_y))-int(np.rint(c_b*scale_y)),int(np.rint(c_l*scale_x)):int(np.rint(self.frame_shape[1]*scale_x))-int(np.rint(c_r*scale_x))], (self.frame_shape[0],self.frame_shape[1]),mode='bilinear')[0].numpy()
        except:
            #Exception handling is implemented to prevent the process from stopping. However, if the keypoints are not broken, this exception block should never be triggered. It's recommended to detect broken keypoints in your data beforehand. The exception will be thrown if cropping results in a size of 0 in any dimension. We are quite certain that this exception block was never encountered when training our model ablations from the paper. 
            driving_augm3_crop = torch.nn.functional.interpolate(torch.from_numpy(driving_augm3_crop[None]), (self.frame_shape[0],self.frame_shape[1]),mode='bilinear')[0].numpy()
            print('Could not crop')
            
        result = {
            'input_images_augm':            input_images_augm,
            'input_images_augm2':           input_images_augm2,         
            'input_images_augm2_uncropped': input_images_augm2_uncropped,
            'kp_scale':                     kp_scale,
            'kp_shift':                     kp_shift,         
            'input_pos':                    self.idx_grids[:self.num_src],       
            'target_pixels_augm':           target_pixels_augm,     
            'target_pos':                   target_pos,           
            'target_image_augm':            driving_augm,         
            'target_image_augm2':           driving_augm2,              
            'target_image_augm2_uncropped': driving_augm2_uncropped,
            'target_image_augm3_crop':      driving_augm3_crop,   
            'remaining_pos':                remaining_pos,
            'sampled_idxs':                 sampled_idxs,
            'remaining_idxs':               remaining_idxs,
            'sceneid':                      idx,                  
        }


        return result



