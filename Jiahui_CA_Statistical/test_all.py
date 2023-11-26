import os
import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model.networks import Generator
from utils.tools import get_config, random_bbox, mask_image, is_image_file, default_loader, normalize, get_model_list

from data.dataset import Dataset

import utils.losses

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--image', type=str)
parser.add_argument('--mask', type=str, default='')
parser.add_argument('--output', type=str, default='output.png')
parser.add_argument('--flow', type=str, default='')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--iter', type=int, default=0)

def main():
    args = parser.parse_args()
    config = get_config(args.config)


    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    print("Arguments: {}".format(args))

    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    print("Configuration: {}".format(config))

    try:  # for unexpected error logging
        with torch.no_grad(): 
            if os.path.exists(config['test_in_dir']):

                test_dataset = Dataset(data_path=config['test_in_dir'],
                            with_subfolder=config['data_with_subfolder'],
                            image_shape=config['image_shape'],
                            random_crop=config['random_crop'])
                print("###################################################")
                test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                        batch_size=config['batch_size'],
                                                        shuffle=False,
                                                        num_workers=config['num_workers'])
                print("###################################################")

                # Set checkpoint path
                if not args.checkpoint_path:
                    checkpoint_path = os.path.join('checkpoints',
                                                   config['dataset_name'],
                                                   config['mask_type'] + '_' + config['expname'])
                else:
                    checkpoint_path = args.checkpoint_path

                # Define the trainer
                netG = Generator(config['netG'], cuda, device_ids)
                # Resume weight
                last_model_name = get_model_list(checkpoint_path, "gen", iteration=args.iter)
                netG.load_state_dict(torch.load(last_model_name))
                model_iteration = int(last_model_name[-11:-3])
                print("Resume from {} at iteration {}".format(checkpoint_path, model_iteration))

                iterable_test_loader = iter(test_loader)
                batch_num = 0

                # orig_imgs, out_imgs, img_sz=(256, 256)
                orig_imgs = []
                out_imgs = []
                img_sz = config['image_shape']

                while True:
                    try:
                        ground_truth = next(iterable_test_loader)
                    except StopIteration:
                        break

                    # Prepare the inputs
                    bboxes = random_bbox(config, batch_size=ground_truth.size(0))
                    x, mask = mask_image(ground_truth, bboxes, config)
                    
                    if cuda:
                        netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
                        x = x.cuda()
                        mask = mask.cuda()
                        netG = netG.to("cuda")
                        ground_truth = ground_truth.cuda()

                    # Inference
                    x1, x2, offset_flow = netG(x, mask)
                    inpainted_result = x2 * mask + x * (1. - mask)

                    for i in range(ground_truth.shape[0]):
                        # viz_max_out = config['viz_max_out']
                        if config['test_img_save']:
                            viz_images = torch.stack([ground_truth[i:i+1], x[i:i+1], inpainted_result[i:i+1]], dim=1)
                
                            viz_images = viz_images.view(-1, *list(x.size())[1:])
                            vutils.save_image(viz_images,
                                        '%s/niter_%03d_%03d.png' % (config['test_out_dir'], batch_num,i),
                                        nrow=3 * 4,
                                        normalize=True)
                        if config['compute_loss']:
                            orig_imgs.append(ground_truth[i].cpu().numpy())
                            out_imgs.append(inpainted_result[i].cpu().numpy())
                        

                    batch_num += 1
            
                if config['compute_loss']:
                    for loss in config['test_losses']:
                        loss_fn = getattr(utils.losses, loss, None)
                        score = loss_fn(orig_imgs, out_imgs, img_sz)
                        print("Score for {} is {} using model {}".format(loss_fn.__name__, score, config['mask_type']))



            else:
                raise TypeError("{} is not an image file.".format)
        # exit no grad context
    except Exception as e:  # for unexpected error logging
        print("Error: {}".format(e))
        raise e


if __name__ == '__main__':
    main()
