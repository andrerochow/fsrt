import face_alignment 
import os
import imageio
import numpy as np

from argparse import ArgumentParser
from skimage.transform import resize
from tqdm.auto import tqdm
from scipy.spatial import ConvexHull

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder_in", required=True, help="path to videos")
    parser.add_argument("--folder_out", required=True, help="output folder path")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode")
    opt = parser.parse_args()

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False,
                                      device='cpu' if opt.cpu else 'cuda')
    
    use_ConvexHull = True
    for file in os.listdir(opt.folder_in):
        if file.endswith(".mp4"):
            kps = []
            reader = imageio.get_reader(os.path.join(opt.folder_in,file))
            video = []
            try:
                for im in reader:
                   video.append(im)
            except RuntimeError:
                pass
            reader.close()
            video = [resize(frame, (256, 256))[..., :3] for frame in video]
            for i, image in tqdm(enumerate(video)):
                fa_kps = fa.get_landmarks(255 * image)
                if fa_kps is None:
                    #Store an array containing only zeros, allowing it to be easily identified in the dataloader.
                    fa_kps = [np.zeros((68,2))]
                #If there are multiple detections, we aim to identify the face in the foreground by selecting the one with the largest Convex Hull volume or kp variance.
                if len(fa_kps) > 1:
                    if use_ConvexHull:
                        max_idx = np.stack([ConvexHull(kps).volume for kps in fa_kps],0).argmax()
                    else:
                        max_idx = np.var(np.stack(fa_kps,0),1).mean(-1).argmax()
                        #max_idx = (np.sqrt(np.var(np.stack(fa_kps,0),1)[:,0])*np.sqrt(np.var(np.stack(fa_kps,0),1)[:,1])).argmax()
                    fa_kps = fa_kps[max_idx]

                else:
                    fa_kps = fa_kps[0]
                kps.append(fa_kps)
            if len(kps) > 0:
                kps = np.stack(kps,0)
                np.save(os.path.join(opt.folder_out,file[:-4]+'.npy'),kps)

