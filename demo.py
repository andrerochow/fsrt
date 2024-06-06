import sys
import yaml
from argparse import ArgumentParser
from tqdm.auto import tqdm
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
import ffmpeg
from os.path import splitext
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from scipy.spatial import ConvexHull

from srt.checkpoint import Checkpoint
from srt.utils.visualize import draw_image_with_kp
from modules.keypoint_detector import KPDetector
from modules.expression_encoder import ExpressionEncoder
from srt.model import FSRT


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source.data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial[0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = kp_driving
    
    if use_relative_movement:
        kp_value_diff = (kp_driving - kp_driving_initial)
        kp_value_diff *= adapt_movement_scale
        kp_new = kp_value_diff + kp_source

    return kp_new



def extract_keypoints_and_expression(img, model, kp_detector, cfg, src = False):
    assert kp_detector is not None

    bs, c, h, w = img.shape
    nkp = kp_detector.num_kp
    with torch.no_grad():
        kps, latent_dict = kp_detector(img)
        heatmaps = latent_dict['heatmap'].view(bs,nkp,latent_dict['heatmap'].shape[-2],latent_dict['heatmap'].shape[-1])
        feature_maps = latent_dict['feature_map'].view(bs,latent_dict['feature_map'].shape[-3],latent_dict['feature_map'].shape[-2],latent_dict['feature_map'].shape[-1])
        
    if kps.shape[1] == 1:
        kps = kps.squeeze(1)
    
    expression_vector = model.expression_encoder(feature_maps,heatmaps)
    
    if src:
        expression_vector = expression_vector[None]

    return kps, expression_vector



def forward_model(model, expression_vector_src, keypoints_src, expression_vector_driv, keypoints_driv, img_src, idx_grids, cfg, max_num_pixels, z=None):
    render_kwargs = cfg['model']['decoder_kwargs']
    if len(img_src.shape) < 5:
        img_src = img_src.unsqueeze(1)
    if len(keypoints_src.shape) < 4:
        keypoints_src = keypoints_src.unsqueeze(1)
    
    if z is None:
        z = model.encoder(img_src, keypoints_src, idx_grids[:,:1].repeat(1,img_src.shape[1],1,1,1), expression_vector=expression_vector_src)
    
    target_pos = idx_grids[:,1]
    target_kps = keypoints_driv
    
    _, height, width = target_pos.shape[:3]
    target_pos = target_pos.flatten(1, 2)

    target_kps = target_kps.unsqueeze(1).repeat(1, target_pos.shape[1], 1,1)
    
    num_pixels = target_pos.shape[1]
    img = torch.zeros((target_pos.shape[0],target_pos.shape[1],3))

    for i in range(0, num_pixels, max_num_pixels):
        img[:, i:i+max_num_pixels], extras = model.decoder(
            z.clone(), target_pos[:, i:i+max_num_pixels], target_kps[:, i:i+max_num_pixels], expression_vector=expression_vector_driv)

    return img.view(img.shape[0], height, width, 3), z



def make_animation(source_image, driving_video, model, kp_detector, cfg, max_num_pixels, relative=False, adapt_movement_scale=False):
    _, y, x= np.meshgrid(np.zeros(2),np.arange(source_image.shape[-3]), np.arange(source_image.shape[-2]), indexing="ij")
    idx_grids = np.stack([x, y], axis=-1).astype(np.float32)
    #Normalize
    idx_grids[...,0] = (idx_grids[...,0]+0.5 -((source_image.shape[-3])/2.0))/((source_image.shape[-3])/2.0)
    idx_grids[...,1] = (idx_grids[...,1]+0.5 -((source_image.shape[-2])/2.0))/((source_image.shape[-2])/2.0)
    idx_grids = torch.from_numpy(idx_grids).cuda().unsqueeze(0)
    z = None
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image.astype(np.float32)).permute(0, 3, 1, 2).cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source, expression_vector_src = extract_keypoints_and_expression(source.clone(), model, kp_detector,cfg, src=True)
        kp_driving_initial, _ = extract_keypoints_and_expression(driving[:, :, 0].cuda().clone(), model, kp_detector,cfg)

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx].cuda()
            kp_driving, expression_vector_driv = extract_keypoints_and_expression(driving_frame.clone(), model, kp_detector,cfg)
            
            kp_norm = normalize_kp(kp_source=kp_source[0], kp_driving=kp_driving,
                                kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                adapt_movement_scale=adapt_movement_scale)
                
            out, z =  forward_model(model,expression_vector_src, kp_source, expression_vector_driv, kp_norm, source.unsqueeze(0), idx_grids, cfg, max_num_pixels, z=z)
            #img_kp = torch.from_numpy(draw_image_with_kp(torch.clamp(out[0],0.,1.).cpu().numpy(),kp_norm['kp'][0].cpu().numpy()))
            predictions.append(torch.cat([driving_frame.detach()[0].permute(1,2,0).cpu(),torch.clamp(out[0],0.,1.)],dim=-2))
    return predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment 
    from scipy.spatial import ConvexHull

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source[0])[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", required=True, help="path to checkpoint to restore")
    parser.add_argument("--source_image", required=True, help="path to source (image or mp4)")
    parser.add_argument("--driving_video", default='driving.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source. (requires face_aligment lib)")
    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None, help="Set frame to start from.")
    parser.add_argument('--source_idx', nargs='+', help='Indices of the source images in the source video (e.g. 0 10 -1 for idx 0, idx 10, idx -1)', default='0')
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode (only for FaceAlignment keypoint extraction).")
    parser.add_argument("--audio", dest="audio", action="store_true", help="copy audio to output from the driving video" )
    parser.add_argument("--max_num_pixels", default=65536, help="number of parallel processed pixels. Reduce this value if you run out of GPU memory!")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)
    parser.set_defaults(audio_on=False)

    opt = parser.parse_args()
    opt.source_idx = [int(i) for i in opt.source_idx]
    if opt.source_image[-4:] == '.mp4':
        reader = imageio.get_reader(opt.source_image)
        source_image = []
        try:
            for im in reader:
                source_image.append(im)
        except RuntimeError:
            pass
        reader.close()
    else:
        source_image = [imageio.imread(opt.source_image)]
    source_image = [source_image[opt.source_idx[i]] for i in range(len(opt.source_idx))]
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()
    
    with open(opt.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.CLoader)
        
    kp_detector = KPDetector().cuda()
    kp_detector.load_state_dict(torch.load('./weights/kp_detector.pt'))
    expression_encoder = ExpressionEncoder(expression_size=cfg['model']['expression_size'], in_channels=kp_detector.predictor.out_filters) 
        
    model = FSRT(cfg['model'],expression_encoder=expression_encoder).cuda()
    
    model.eval()
    kp_detector.eval()
    
    encoder_module = model.encoder
    decoder_module = model.decoder
    expression_encoder_module = model.expression_encoder
    

    source_image = [resize(img, (256, 256))[..., :3] for img in source_image]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    source_image = np.array(source_image)
    
    #Load the checkpoints
    checkpoint = Checkpoint('./', device='cuda:0', encoder=encoder_module,
                                decoder=decoder_module, expression_encoder=expression_encoder_module)
    load_dict = checkpoint.load(opt.checkpoint)    


    if opt.find_best_frame or opt.best_frame is not None:
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, model, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cfg=cfg, max_num_pixels=opt.max_num_pixels)
        predictions_backward = make_animation(source_image, driving_backward, model, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cfg=cfg, max_num_pixels=opt.max_num_pixels)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, model, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cfg=cfg, max_num_pixels=opt.max_num_pixels)
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=20)
#    imageio.mimsave(opt.result_video, [img_as_ubyte(np.concatenate([source_image[0],frame], axis=1)) for frame in predictions], fps=20)
    if opt.audio:
        with NamedTemporaryFile(suffix='.' + splitext(opt.result_video)[1]) as output:
            ffmpeg.output(ffmpeg.input(opt.result_video).video, ffmpeg.input(opt.driving_video).audio, output.name, c='copy').run()
            with open(opt.result_video, 'wb') as result:
                copyfileobj(output, result) 
