import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import os
import imageio
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.util import normalize_kp
from skimage import img_as_float32, img_as_ubyte
from imageio import imread

def load_checkpoints(config_path, checkpoint_detector_path, checkpoint_generator_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)
    
    if cpu:
        detector = torch.load(checkpoint_detector_path, map_location=torch.device('cpu'))
        generator = torch.load(checkpoint_generator_path, map_location=torch.device('cpu'))
    else:
        detector = torch.load(checkpoint_detector_path)
        generator = torch.load(checkpoint_generator_path)
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        detector = DataParallelWithCallback(detector)

    generator.eval()
    detector.eval()
    
    return generator, detector, config

def make_animation(source_image, driving_video, generator, detector, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = detector(source)
        kp_driving_initial = detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=True,
                                   use_relative_jacobian=True, adapt_movement_scale=False)
            # out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            out = generator(source, kp_source, kp_norm)
            out['kp_source'] = kp_source
            out['kp_driving'] = kp_driving
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

    return predictions

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='config/pelvic-256.yaml', help="path to config")
    parser.add_argument("--checkpoint_detector", default='checkpoints/pelvic_detector.pth.tar', help="detector path to checkpoint to restore")
    parser.add_argument("--checkpoint_generator", default='checkpoints/pelvic_generator.pth.tar', help="generator path to checkpoint to restore")
    parser.add_argument("--source_path", default='demo_img/source/0063.png', help="path to source image")
    parser.add_argument("--driving_path", default='demo_img/driving/0041.mp4', help="path to driving video")
    parser.add_argument("--result_path", default='result_test.mp4', help="path to output")
    parser.add_argument("--result_combine_path", default='result_combine_test.mp4', help="path to output")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.set_defaults(cpu=False)
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # driving video
    reader = imageio.get_reader(opt.driving_path)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            im = rgb2gray(im)
            im = np.expand_dims(im, -1)
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    driving_video = [resize(frame, (256, 256)) for frame in driving_video]
    length = len(driving_video)

    # source image
    source_image = img_as_float32(imread(opt.source_path))
    source_image = resize(source_image, (256, 256))[..., np.newaxis]

    # load pretrained model
    generator, kp_detector, config = load_checkpoints(config_path=opt.config, checkpoint_detector_path=opt.checkpoint_detector, checkpoint_generator_path=opt.checkpoint_generator, cpu=opt.cpu)

    # inference
    mid_frame = length // 2
    driving_forward = driving_video[mid_frame:]
    driving_backward = driving_video[:(mid_frame+1)][::-1]
    predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, cpu=opt.cpu)
    predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, cpu=opt.cpu)
    predictions = predictions_backward[::-1] + predictions_forward[1:]
    fps = 10
    #  save prediction video
    imageio.mimsave(opt.result_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

    combine_video = []
    # combine driving and prediction
    for driving_frame, predict_frame in zip(driving_video, predictions):
        combine_frame = np.concatenate((driving_frame, predict_frame), axis=1)
        combine_video.append(img_as_ubyte(combine_frame))

    #  save driving and prediction video
    imageio.mimsave(opt.result_combine_path, [img_as_ubyte(frame) for frame in combine_video], fps=fps)    