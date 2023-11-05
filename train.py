import argparse
import logging
import sys
import os 
from pathlib import Path
import collections
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import PlantDataset, PlantDatasetCameraFrame
from utils.metrics import dice_loss
from evaluate import evaluate
from unet.unet_model import UNet, HeightMapOnly, weights_init, load_model
import matplotlib.pyplot as plt
import utils.utils as unet_utils
import pdb

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--wandb_project_name', type=str, default='plant_new')
    parser.add_argument('--data_df_path', type=str, required=True)
    parser.add_argument('--epochs', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--evaluation_interval', type=int, default=200, help='Number of epochs')
    parser.add_argument('--evaluate_train', action='store_true')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    
    parser.add_argument('--evaluate_validation', action='store_true', default=False, help='Only evaluate.')
    parser.add_argument('--evaluate_all_model_in_this_folder', type=str, help='Evaluate every model in the folder.')
    parser.add_argument('--evaluate_all_model_start', type=int, default=0)
    parser.add_argument('--evaluate_all_model_end', type=int, default=20000)
    parser.add_argument('--evaluate_num_to_plot', type=int, default=5)
    parser.add_argument('--test_set', default='Val', help='Val|Test')
    parser.add_argument('--experiment_name', type=str, help='Experiment to add to the folder name that saves model and images. If evaluate_validation, and experiment_name is not provided, no image will be saved')
    parser.add_argument('--paper_figures', action='store_true')

    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--weights_prefix_to_load', type=str, nargs='+', default=[])
    
    parser.add_argument('--dir_checkpoint', '-d', type=str, help='Folder to save images and model.')
    
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode, no logging with wandb, no saving figures.')
    parser.add_argument('--disable_wandb', action='store_true', default=False, help='Save to local folder, do not log with wandb.')
    
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--percentage_of_train_data', type=float, default=1.0)
    parser.add_argument('--train_indices_file', type=str, help="Order of the train indices to determine which ones are used for training")
    parser.add_argument('--classes', nargs='+', type=float, default=[-0.03], help='Thresholds of amount of height decrease')
    parser.add_argument('--class_freq', nargs='+', type=float, help='Weights to input into the cross entropy loss for imbalanced data.')
    parser.add_argument('--threshold_class', type=int, default=1)
    parser.add_argument('--close_to_board',  action='store_true')
    # parser.add_argument('--no_action_stream', action='store_true', help='Do not use separate linear/conv layers to process action input.')
   
    parser.add_argument('--predict_precondition', action='store_true')
    parser.add_argument('--precondition_input_dim', type=int, default=1)
    parser.add_argument('--no_action', action='store_true', help='Do not condition on action.')
    parser.add_argument('--no_image', action='store_true', help='Do not condition on image.')
    parser.add_argument('--use_resnet', action='store_true', help='Use resnet as encoder')
    parser.add_argument('--linear_comb_heightmap', action='store_true', help='1x1 conv of the height map')
    parser.add_argument('--no_rgb_feature', action='store_true', help='Still use the original UNet, but do not use rgb feature')

    parser.add_argument('--resnet_maxpool_after_enc', action='store_true', help="When using resnet, keep the maxpool after the first conv layer")
    parser.add_argument('--freeze_resnet', action='store_true', help='Freeze the 4 layers after the first convolution.')
    parser.add_argument('--action_as_image', action='store_true', help='Take in action as an image.')
    parser.add_argument('--unet_downsample_layer_type', type=str, default="DownMaxpoolSingleConv", help='e.g. DownRelu, DownMaxpoolSingleConv')
    parser.add_argument('--downconv_channels', nargs='+', type=int, default=[64, 64, 128, 256, 512])
    parser.add_argument('--upconv_channels', nargs='+', type=int, default=[512, 512, 256, 256, 128, 128, 64, 64])
    parser.add_argument('--action_input_dim', type=int, default=9)
    parser.add_argument('--distance_to_boundary', type=float)
    parser.add_argument('--distance_to_boundary_in_input_image', type=float, default=0.2)
    parser.add_argument('--action_upconv_channels', nargs='+', type=int, default=[64])
    parser.add_argument('--action_upconv_sizes', nargs='+', type=int)
    
    parser.add_argument('--rotation_augmentation', action='store_true')
    parser.add_argument('--remove_far_background', action='store_true')
    parser.add_argument('--remove_far_background_all', action='store_true')
    parser.add_argument('--use_height_feature', action='store_true')
    parser.add_argument('--no_background', action='store_true')

    parser.add_argument('--height_map_min_value', type=float, default=-0.05)
    parser.add_argument('--height_map_value_range', type=float, default=0.25)
    parser.add_argument('--height_diff_clip_range', nargs='+', type=float, default=[-0.1,0.1])
    # parser.add_argument('--coordinate_feature_range', nargs='+', type=int, default=[464])
    
    parser.add_argument('--report_multi_class_ap', action='store_true')
    parser.add_argument('--use_regression_loss', action='store_true')
    parser.add_argument('--use_nll_loss', action='store_true')
    parser.add_argument('--huber_loss_delta', type=float)
    parser.add_argument('--use_postaction_mask_loss', action='store_true')
    
    parser.add_argument('--no_color_jitter', action='store_true')
    parser.add_argument('--no_flipping', action='store_true')
    parser.add_argument('--color_jitter_setting', nargs='+', type=float, default=[0.5,0.5,0.5,0.5])

    
    parser.add_argument('--dataset_from_camera_frame', action='store_true')
    parser.add_argument('--crop_size', type=int)
    parser.add_argument('--input_image_size', type=int)
    parser.add_argument('--minimum_decrease_before_transformation', type=float)

    return parser.parse_args()


def train_net(
    args,
    df,
    net,
    device,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    amp: bool = True,
):
    data_set_type = PlantDatasetCameraFrame if args.dataset_from_camera_frame else PlantDataset
    n_classes = net.n_classes
    if args.use_regression_loss:
        n_classes -= 1
    colormap_colors = unet_utils.create_colormap(n_classes, args.threshold_class)
    threshold_class = args.threshold_class
    
    # 1. Create dataset 
    train_df = df[df['split'] == "Train"]
    if args.train_indices_file:
        train_indices_random_order = np.load(args.train_indices_file)
        train_length = int(len(train_df) * args.percentage_of_train_data)
        train_df = train_df.iloc[list(train_indices_random_order[:train_length])]
    train_df = train_df.reset_index(drop=True)
    train_set = data_set_type(train_df, "train", vars(args))
    val_df = df[df['split'] == "Val"]
    val_df = val_df.reset_index(drop=True)
    val_set = data_set_type(val_df, "val", vars(args))
    n_train = len(train_set)
    n_val = len(val_set)

    # 2. Create data loaders
    num_workers = 0 if args.debug else 4
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    # (Initialize logging)
    if not args.debug:
        if args.disable_wandb:
            import datetime
            import time 
            ts = time.time()                                                                                            
            run_name = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S') 
        else:
            experiment = wandb.init(project=args.wandb_project_name, resume='allow', anonymous='must', config=vars(args))
            run_name = wandb.run.name 
        if args.experiment_name is not None:
            run_name = f"{run_name}-{args.experiment_name}"
        dir_experiment_checkpoint = os.path.join(args.dir_checkpoint, run_name)
        Path(dir_experiment_checkpoint).mkdir(parents=True, exist_ok=True)
    print(net)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')
    
    # 3. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    if args.class_freq is not None and len(args.class_freq) == len(args.classes)+1:
        ce_weights = [1/x for x in args.class_freq]
        weights = torch.tensor(ce_weights)
        weights = weights.to(device=device)
        criterion = nn.CrossEntropyLoss(reduction='none', weight=weights)
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')
    
    if len(args.classes) > 1 and args.use_nll_loss:
        logging.info("Use NLLLoss")
        criterion_nll = nn.NLLLoss(reduction='none')
    if args.use_regression_loss:
        if args.huber_loss_delta:
            criterion_reg_loss_aux = nn.HuberLoss(reduction='none', delta=args.huber_loss_delta)
        else:
            criterion_reg_loss_aux = nn.MSELoss(reduction='none')
    if args.use_postaction_mask_loss:
        criterion3 = nn.CrossEntropyLoss(reduction='none')
    global_step = 0
    total_samples = 0
    # 4. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        train_loader.dataset.reset()
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                action_info = batch['action']
                image_loss_mask = 1 - batch['image_mask'].float()

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                action_info = action_info.to(device=device, dtype=torch.float32)
                image_loss_mask = image_loss_mask.to(device=device, dtype=torch.float32)

                loss_dict = {}
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images, action_info)
                    
                    if args.predict_precondition:
                        precond_gt = batch['precond_gt'].to(device=device, dtype=torch.long)
                        loss_all = criterion(masks_pred, precond_gt) 
                        loss = loss_all.sum() / len(loss_all)
                        
                    else:
                        # seperate out the last channel for using regression loss
                        if args.use_regression_loss:
                            masks_pred_discrete = masks_pred[:,:-1,:,:]
                            masks_pred_cont = masks_pred[:,-1,:,:]
                        else:
                            masks_pred_discrete = masks_pred
                        if args.use_postaction_mask_loss:
                            masks_pred_postaction_mask = masks_pred_discrete[:,-2:,:,:]
                            masks_pred_discrete = masks_pred_discrete[:,:-2,:,:]
                        
                        loss_all = criterion(masks_pred_discrete, true_masks) * image_loss_mask 
                        loss = loss_all.sum() / image_loss_mask.sum() 
                        loss_dict['train class loss'] = loss.item()
                        
                        if len(args.classes) > 1 and args.use_nll_loss:
                            mask_true_one_hot = F.one_hot(true_masks, len(args.classes)+1).permute(0, 3, 1, 2)
                            mask_pred_softmax = F.softmax(masks_pred_discrete, dim=1)
                            background_class_lprob = torch.log(torch.sum(mask_pred_softmax[:,:threshold_class,...], dim=1))
                            height_decrease_class_lprob  = torch.log(torch.sum(mask_pred_softmax[:,threshold_class:,...], dim=1))
                            true_masks_2class = torch.clone(true_masks)
                            true_masks_2class[true_masks_2class < threshold_class] = 0
                            true_masks_2class[true_masks_2class >= threshold_class] = 1
                            loss_background_foreground_all = criterion_nll(torch.stack([background_class_lprob, height_decrease_class_lprob], dim=1), true_masks_2class) * image_loss_mask
                            loss_background_foreground = loss_background_foreground_all.sum() / image_loss_mask.sum() 
                            loss_dict['train 2 class loss'] = loss_background_foreground.item()
                            loss += loss_background_foreground

                        if args.use_regression_loss:
                            height_diff = batch['height_diff'].to(device=device, dtype=torch.float32)
                            reg_loss = criterion_reg_loss_aux(masks_pred_cont, height_diff) * image_loss_mask 
                            reg_loss = reg_loss.sum() / image_loss_mask.sum() 
                            loss_dict['train reg loss'] = reg_loss.item()
                            loss += reg_loss
                        
                        if args.use_postaction_mask_loss:
                            postaction_mask = batch['postaction_mask']
                            postaction_mask = postaction_mask.to(device=device, dtype=torch.long)
                            postaction_mask_loss_all = criterion(masks_pred_postaction_mask, postaction_mask) * image_loss_mask 
                            postaction_mask_loss = postaction_mask_loss_all.sum() / image_loss_mask.sum() 
                            loss += postaction_mask_loss

                    loss_dict['train loss'] = loss.item()

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                wandb_dict = {}
                if not args.debug:
                    if not args.disable_wandb:
                        wandb_dict.update(loss_dict)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                if global_step % args.evaluation_interval == 0: # 
                    # if not args.debug:
                    #     histograms = {}
                    #     for tag, value in net.named_parameters():
                    #         tag = tag.replace('/', '.')
                    #         if not torch.isinf(value).any():
                    #             histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                    #         if not torch.isinf(value.grad).any():
                    #             histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    def update_wandb_dict_with_fig_dict(fig_dict, split):
                        for fig_k, fig in fig_dict.items():
                            if fig is None:
                                continue 

                            # if not args.disable_wandb:
                            #     final_img = unet_utils.plt_to_image(fig)
                            #     wandb_dict.update({f"{split}_image_{fig_k}": wandb.Image(final_img)})
                            # else:
                            image_path = os.path.join(dir_experiment_checkpoint,f"{split}_image_{fig_k}_epoch{epoch}_step{global_step}.png")
                            final_img = unet_utils.plt_to_image(fig)
                            final_img.save(image_path) 
                            
                    fig_dict, metric_dict_val = evaluate(net, val_loader, vars(args), device, 'validation', colormap_colors, num_batches_to_plot = 10, num_to_plot_in_each_batch = 10)
                    
                    if not args.debug:
                        if not args.disable_wandb:
                            wandb_dict.update({
                                'learning rate': optimizer.param_groups[0]['lr'],
                            })
                            for k,v in metric_dict_val.items():
                                wandb_dict[f'val_{k}'] = v
                        update_wandb_dict_with_fig_dict(fig_dict, "evaluate")
                            
                    plt.close('all') 
                    if args.evaluate_train:
                        fig_train_dict, metric_dict_train = evaluate(net, train_loader, vars(args), device, 'train', colormap_colors, num_batches_to_plot = 3, num_to_plot_in_each_batch = 5)
                        if not args.debug:
                            if not args.disable_wandb:
                                for k,v in metric_dict_train.items():
                                    wandb_dict[f'train_{k}'] = v
                            update_wandb_dict_with_fig_dict(fig_train_dict, "train")
                        plt.close('all') 

                    if not args.debug:
                        torch.save(net.state_dict(), os.path.join(dir_experiment_checkpoint, 'checkpoint_step_{}.pth'.format(global_step)))
                        logging.info(f'Checkpoint step {global_step} saved!')
                
                if not args.debug and not args.disable_wandb:
                    experiment.log(wandb_dict, step = global_step)

def evaluate_all_model_in_folder(hp):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder = hp['evaluate_all_model_in_this_folder']
    step_to_model_path = {}
    all_ateps = []
    for file in os.listdir(folder):
        if not file.endswith(".pth"):
            continue
        step = int(file.split(".")[0].split("_")[2])
        if step >= hp['evaluate_all_model_start'] and step <= hp['evaluate_all_model_end']:
            all_ateps.append(step)
        step_to_model_path[step] = os.path.join(folder, file)
    all_ateps = sorted(all_ateps)  
    df = pd.read_csv(hp['data_df_path'])
    df_split = df[df['split'] == hp['test_set']].reset_index(drop=True)
    val_set = PlantDatasetCameraFrame(df_split, hp['test_set'], hp) if hp['dataset_from_camera_frame'] else PlantDataset(df_split, hp['test_set'], hp) 
    loader_args = dict(batch_size=hp['batch_size'], num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    net = UNet(hp)        
    n_classes = net.n_classes
    if hp['use_regression_loss']:
        n_classes -= 1
    colormap_colors = unet_utils.create_colormap(n_classes, args.threshold_class)
    
    net.to(device=device) 
    if not hp['predict_precondition']:
        iou_result_0 = collections.defaultdict(list)
        iou_result_1 = collections.defaultdict(list)
        for step in all_ateps:
            print("Loading: ", step_to_model_path[step])
            net.load_state_dict(torch.load(step_to_model_path[step], map_location=device))
            
            
            for iou_threshold in np.arange(0.05,1,0.05):
                fig_dict, metric_dict_val = evaluate(net, val_loader, hp, device, args.test_set.lower(), colormap_colors, num_batches_to_plot = 0, num_to_plot_in_each_batch = 3, do_not_report_ap=True, iou_threshold=iou_threshold)
                iou_result_0[iou_threshold].append(metric_dict_val['iou_0'])
                iou_result_1[iou_threshold].append(metric_dict_val['iou_1'])
                 
        import pickle
        with open(os.path.join(folder, "iou.pkl"), "wb+") as f:
            pickle.dump([all_ateps, dict(iou_result_0), dict(iou_result_1)], f)
    else:
        metric_all = collections.defaultdict(list)
        for step in all_ateps:
            print("Loading: ", step_to_model_path[step])
            net.load_state_dict(torch.load(step_to_model_path[step], map_location=device))
            fig_dict, metric_dict_val = evaluate(net, val_loader, hp, device, args.test_set.lower(), colormap_colors, num_batches_to_plot = 0, num_to_plot_in_each_batch = 3, do_not_report_ap=True)
            for k,v in metric_dict_val.items():
                metric_all[k].append(v)
            plt.close('all')
        import pickle
        with open(os.path.join(folder, "roc.pkl"), "wb+") as f:
            pickle.dump([all_ateps, metric_all], f)

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if args.close_to_board:
        logging.info("Mask only shows spaces that are close to board, so setting threshold class to 1.")
        logging.info("WARNING: no args.classes is used.")
        args.classes = [-0.03]
        args.threshold_class = 1
    if args.evaluate_all_model_in_this_folder is not None:
        evaluate_all_model_in_folder(vars(args))
        exit(0)

    if args.linear_comb_heightmap:
        assert args.use_height_feature
        net = HeightMapOnly(vars(args))
        net.apply(lambda x : weights_init(x, 0.02))
    else:
        net = UNet(vars(args))
        net.apply(lambda x : weights_init(x, 0.02))
    for name,param in net.named_parameters():
        if param.requires_grad == False:
            print(":: Not learning: ", name)
    if args.load:
        net = load_model(net, args.load, args.weights_prefix_to_load)
        #load_dict = torch.load(args.load, map_location=device)
        #net.load_state_dict(load_dict)
        logging.info(f':: Model loaded from {args.load}')

    
    net.to(device=device)
    df = pd.read_csv(args.data_df_path)
    data_set_type = PlantDatasetCameraFrame if args.dataset_from_camera_frame else PlantDataset
    if args.evaluate_validation:
        n_classes = net.n_classes
        if args.use_regression_loss:
            n_classes -= 1
        colormap_colors = unet_utils.create_colormap(n_classes, args.threshold_class)
        
        df_split = df[df['split'] == args.test_set]
        df_split = df_split.reset_index(drop=True)
        val_set = data_set_type(df_split, "val", vars(args))
        loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
        hp = vars(args)
        if args.experiment_name is not None:
            import datetime
            import time 
            ts = time.time()                                                                                            
            run_name = datetime.datetime.fromtimestamp(ts).strftime('evaluate_validation_%Y_%m_%d_%H_%M_%S') 
            run_name = f"{run_name}-{args.experiment_name}"
            dir_experiment_checkpoint = os.path.join(args.dir_checkpoint, run_name)
            Path(dir_experiment_checkpoint).mkdir(parents=True, exist_ok=True)
        
            hp['dir_experiment_checkpoint'] = dir_experiment_checkpoint
        fig_dict, metric_dict_val = evaluate(net, val_loader, hp, device, args.test_set.lower(), colormap_colors, num_batches_to_plot = args.evaluate_num_to_plot, num_to_plot_in_each_batch = 30, iou_threshold=args.iou_threshold)
        log_str = ""
        for k,v in metric_dict_val.items():
            log_str += f" {k} : {v:.3f} |"
        logging.info(f'Evaluate {args.test_set}: ' + log_str)
        if args.experiment_name is not None:
            for fig_k, fig in fig_dict.items():
                if fig is None:
                    continue 
                image_path = os.path.join(dir_experiment_checkpoint,f"evaluate_image_{fig_k}.png")
                plt.savefig(image_path, bbox_inches = 'tight', pad_inches = 0)
                # final_img = unet_utils.plt_to_image(fig)
                # final_img.save(image_path) 
        exit(0)

    train_net(
        args, 
        df,
        net=net,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        amp=True,
    )
