import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
import utils.metrics as metrics_utils
import utils.utils as unet_utils
import collections
import os
import matplotlib.pyplot as plt

def evaluate(net, dataloader, hp, device, split, colormap_colors, num_batches_to_plot = 10, num_to_plot_in_each_batch = 3, do_not_report_ap=False, iou_threshold=0.5):
    '''
    do_not_report_ap: when calculating iou's for different threshold, do not calcualte ap to reduce computation time.
    '''
    net.eval()
    num_val_batches = len(dataloader)
    metric_dict = collections.defaultdict(list)
    fig_dict = collections.defaultdict(lambda : None)
    criterion = nn.CrossEntropyLoss(reduction='none')
    num_plots_per_row = 5
    threshold_class = hp['threshold_class']
    n_classes = net.n_classes
    if hp['use_regression_loss']:
        if hp['huber_loss_delta']:
            criterion_reg_loss_aux = nn.HuberLoss(reduction='none', delta=hp['huber_loss_delta'])
        else:
            criterion_reg_loss_aux = nn.MSELoss(reduction='none')
        n_classes -= 1
        num_plots_per_row += 2
    img_height, img_width = dataloader.dataset.img_height, dataloader.dataset.img_width
    # plotting 
    plot_batches = np.random.choice(num_val_batches, min(num_val_batches, num_batches_to_plot), replace=False)
    paper_figures = hp['paper_figures']
    # fig, axes_iterator = unet_utils.create_axes(num_to_plot_in_each_batch * len(plot_batches) * 4, 4)
    
    # iterate over the validation set
    np.random.seed(1119)
    # print(num_val_batches)
    pbar = tqdm(dataloader, total=num_val_batches, desc=f'{split} round', unit='batch', leave=False)

    batch_idx = 0
    num_samples = 0
    
    # for plotting precision-recall
    all_pred = collections.defaultdict(list)
    all_gt = collections.defaultdict(list)
    all_multi_pred = collections.defaultdict(list)
    all_multi_gt = collections.defaultdict(list)
    if hp['predict_precondition']:
        all_precond_pred = collections.defaultdict(list)
        all_precond_gt = collections.defaultdict(list)
    
    for batch in pbar:
        image, mask_true = batch['image'], batch['mask']
        action_info = batch['action']
        image_loss_mask = 1 - batch['image_mask'].float()
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        action_info = action_info.to(device=device, dtype=torch.float32)
        image_loss_mask = image_loss_mask.to(device=device, dtype=torch.float32)
        mask_true_one_hot = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image, action_info)
            if hp['predict_precondition']:
                precond_gt = batch['precond_gt'].to(device=device, dtype=torch.long)
                loss_all = criterion(mask_pred, precond_gt) 
                #loss = loss_all.sum() / len(loss_all) 
                #print(loss_all.shape)
                metric_dict['class loss'] += loss_all.tolist()

                gt_one_hot = F.one_hot(precond_gt, n_classes).float()
                pred_softmax = F.softmax(mask_pred, dim=1).float()
                for cls_idx in [1]:
                    all_precond_pred[cls_idx].append(pred_softmax[:,cls_idx].detach().cpu().numpy())
                    all_precond_gt[cls_idx].append(gt_one_hot[:,cls_idx].detach().cpu().numpy())

                num_samples += len(loss_all)
                batch_idx += 1
                continue

            if hp['use_regression_loss']:
                masks_pred_discrete = mask_pred[:,:-1,:,:]
                masks_pred_cont = mask_pred[:,-1,:,:]
            else:
                masks_pred_discrete = mask_pred
            if hp['use_postaction_mask_loss']:
                masks_pred_postaction_mask = masks_pred_discrete[:,-2:,:,:]
                masks_pred_discrete = masks_pred_discrete[:,:-2,:,:]
                masks_pred_postaction_mask_softmax = F.softmax(masks_pred_postaction_mask, dim=1).float()
            
            loss_all = criterion(masks_pred_discrete, mask_true) * image_loss_mask 
            loss = loss_all.sum(dim=(1,2)) / image_loss_mask.sum(dim=(1,2)) 
            metric_dict['class loss'] += loss.tolist()
            if hp['use_regression_loss']:
                height_diff = batch['height_diff'].to(device=device, dtype=torch.float32)
                reg_loss = criterion_reg_loss_aux(masks_pred_cont, height_diff) * image_loss_mask 
                reg_loss = reg_loss.sum(dim=(1,2)) / image_loss_mask.sum(dim=(1,2)) 
                metric_dict['reg loss'] = reg_loss.tolist()

            # invalid_pixel_mask = batch['image_mask'].to(device=device, dtype=torch.float32)
            # mask_pred_softmax_masked = torch.zeros_like(mask_pred_softmax)
            # mask_pred_softmax_masked[:,0,...] = torch.maximum(mask_pred_softmax[:, 0, ...],invalid_pixel_mask) 
            # mask_pred_softmax_masked[:,1:,...] = mask_pred_softmax[:, 1:, ...] * image_loss_mask.unsqueeze(1)
            
            mask_pred_softmax = F.softmax(masks_pred_discrete, dim=1).float()
            mask_pred_softmax = mask_pred_softmax #* image_loss_mask.unsqueeze(1)
            mask_pred_argmax = masks_pred_discrete.argmax(dim=1)
            
            # mask_pred_one_hot = F.one_hot(mask_pred_argmax, n_classes).permute(0, 3, 1, 2).float()
            # background_class_one_hot = torch.amax(mask_pred_one_hot[:,:threshold_class,...], dim=1)
            # height_decrease_one_hot = torch.amax(mask_pred_one_hot[:,threshold_class:,...], dim=1)
            # class_one_hot = [background_class_one_hot, height_decrease_one_hot]

            if n_classes > 2 and hp['report_multi_class_ap']:
                for cls_idx in range(mask_pred_softmax.shape[1]):
                    ind1,ind2,ind3 = torch.where(image_loss_mask)
                    all_multi_pred[cls_idx].append(mask_pred_softmax[:,cls_idx,...][ind1,ind2,ind3])
                    all_multi_gt[cls_idx].append(mask_true_one_hot[:,cls_idx,...][ind1,ind2,ind3])

                    class_one_hot = metrics_utils.threshold_prob(mask_pred_softmax[:,cls_idx,...], threshold=iou_threshold)
                    metric_dict[f'multi_iou_{cls_idx}'] += metrics_utils.iou_coef(class_one_hot, mask_true_one_hot[:,cls_idx,...], mask=image_loss_mask, per_instance=True).tolist()
            
            background_class_prob = torch.sum(mask_pred_softmax[:,:threshold_class,...], dim=1) # (B, h, w)
            height_decrease_class_prob  = torch.sum(mask_pred_softmax[:,threshold_class:,...], dim=1) # (B, h, w)
            background_class_gt = torch.amax(mask_true_one_hot[:,:threshold_class,...], dim=1)
            height_decrease_gt = torch.amax(mask_true_one_hot[:,threshold_class:,...], dim=1)

            background_class_one_hot = metrics_utils.threshold_prob(background_class_prob, threshold=iou_threshold)
            height_decrease_one_hot  = metrics_utils.threshold_prob(height_decrease_class_prob, threshold=iou_threshold)
            class_one_hot = [background_class_one_hot, height_decrease_one_hot]
            
            image_loss_mask_expanded = image_loss_mask.unsqueeze(1).repeat(1,2,1,1)
            class_prob = [background_class_prob, height_decrease_class_prob]
            class_gt = [background_class_gt, height_decrease_gt]
            
            # for plotting precision-recall curve 
            for cls_idx in range(2):
                ind1,ind2,ind3 = torch.where(image_loss_mask)
                all_pred[cls_idx].append(class_prob[cls_idx][ind1,ind2,ind3])
                all_gt[cls_idx].append(class_gt[cls_idx][ind1,ind2,ind3])
            
                metric_dict[f'iou_{cls_idx}'] += metrics_utils.iou_coef(class_one_hot[cls_idx], class_gt[cls_idx], mask=image_loss_mask, per_instance=True).tolist()
            
            num_samples += len(loss)
            if not batch_idx in plot_batches:
                batch_idx += 1
                continue
            
            if not paper_figures:
                fig, axes_iterator = unet_utils.create_axes(num_to_plot_in_each_batch * num_plots_per_row, num_plots_per_row)
                fig2, axes_iterator2 = unet_utils.create_axes(num_to_plot_in_each_batch * 2, 2)
            plot_indices = np.random.choice(len(mask_true), min(len(mask_true), num_to_plot_in_each_batch), replace=False)
            for i in range(len(plot_indices)):
                idx = plot_indices[i]
                # if i == 0:
                #     print(height_decrease_class_prob[idx])
                
                dataset_idx = batch['dataset_idx'][idx].item()
                end_pixel_start_idx = 7 if hp['action_input_dim'] == 9 else 1
                action = list(batch['action_start_pixel'][idx] * img_width) +  list(batch['action_end_pixel'][idx] * img_width)
                # [x.item() for x in batch['action'][idx][end_pixel_start_idx:] * img_width]
                
                # loss_idx = metric_dict['loss'][num_samples - len(loss) + idx]
                # iou_idx = metric_dict['iou'][num_samples - len(loss) + idx]
                if hp['use_regression_loss']:
                    height_diff_img = batch['height_diff'][idx].numpy()
                    height_diff_pred = masks_pred_cont[idx].detach().cpu().numpy()
                else:
                    height_diff_img = height_diff_pred = None
                
                # if hp['use_regression_loss']:
                #     reg_loss_this = reg_loss.tolist()[idx]
                #     title = f"batch#{batch_idx}_{idx}_dataset#{dataset_idx}_{reg_loss_this:.2f}"
                # else:
                #     title = f"batch#{batch_idx}_{idx}_dataset#{dataset_idx}"
                if not paper_figures:
                    unet_utils.plot_plant_img_and_mask(
                        T.ToPILImage()(batch['image'][idx][:3,:,:]), 
                        batch['image_mask'][idx].numpy(), 
                        batch['depth_img'][idx].numpy(),
                        batch['mask'][idx].numpy(), 
                        mask_pred_argmax[idx].detach().cpu().numpy(),
                        height_decrease_class_prob[idx].detach().cpu().numpy(),
                        height_diff_img, 
                        height_diff_pred,
                        action, 
                        axes_iterator,
                        colormap_colors,
                        title = f"batch#{batch_idx}_{idx}_dataset#{dataset_idx}",
                        num_classes = mask_pred_softmax.shape[1],
                    )
                    if hp['use_postaction_mask_loss']:
                        unet_utils.plot_postactionmask(batch['postaction_mask'][idx].numpy(), masks_pred_postaction_mask_softmax[idx][1].detach().cpu().numpy(), action, axes_iterator2, title = None)
                else:
                    info = {
                        'action_start_pixels' : action[:2],
                        'action_end_pixels' : action[2:],
                        'rgb_image_input' : T.ToPILImage()(batch['image'][idx][:3,:,:]),
                        'height_image_input' : batch['depth_img'][idx].numpy(),
                        'gt' : batch['mask'][idx].numpy(),
                        'pred' : height_decrease_one_hot[idx].detach().cpu().numpy(),
                        'prob' : height_decrease_class_prob[idx].detach().cpu().numpy(),
                        'save_folder' : hp['dir_experiment_checkpoint'],
                        'save_path_prefix' : f'evaluate_image_mask_batch{batch_idx}_{dataset_idx}',
                    }
                    unet_utils.visualize_forward_model_outputs_save_separately(info)
                    
                    # fig_k = f'mask_batch{batch_idx}_{dataset_idx}'
                    # image_path = os.path.join(hp['dir_experiment_checkpoint'],f"evaluate_image_{fig_k}.png")
                    # plt.savefig(image_path, bbox_inches = 'tight', pad_inches = 0)
                    plt.close("all")
            if not paper_figures:
                if 'dir_experiment_checkpoint' in hp:
                    image_path = os.path.join(hp['dir_experiment_checkpoint'],f"evaluate_image_mask_batch{batch_idx}.png")
                    final_img = unet_utils.plt_to_image(fig)
                    final_img.save(image_path) 

                    if hp['use_postaction_mask_loss']:
                        image_path = os.path.join(hp['dir_experiment_checkpoint'],f"evaluate_image_postaction_mask_batch{batch_idx}.png")
                        final_img = unet_utils.plt_to_image(fig2)
                        final_img.save(image_path) 
                    plt.close("all")
                else:
                    fig_dict[f'mask_batch{batch_idx}'] = fig 
                    if hp['use_postaction_mask_loss']:
                        fig_dict[f'postaction_mask_batch{batch_idx}'] = fig2 
        batch_idx += 1
    
    net.train()
    pbar.close()

    for k,v in metric_dict.items():
        metric_dict[k] = sum(v) / num_samples
    if not do_not_report_ap and not hp['predict_precondition']:
        for ch_idx in all_pred.keys():
            # for plotting precision-recall
            all_pred_ch = torch.cat(all_pred[ch_idx],dim=0).detach().cpu().numpy()
            all_gt_ch = torch.cat(all_gt[ch_idx], dim=0).detach().cpu().numpy()
            precision, recall,_ = metrics_utils.precision_recall(all_pred_ch, all_gt_ch)
            score = metrics_utils.map_score(all_pred_ch, all_gt_ch)
            fig_pr_curve = unet_utils.plot_precision_recall_curve(precision, recall)

            fig_dict[f'pr_curve_{ch_idx}'] = fig_pr_curve
            metric_dict[f'map_{ch_idx}'] = score
        if n_classes > 2 and hp['report_multi_class_ap']:
            for ch_idx in all_multi_pred.keys():
                all_pred_ch = torch.cat(all_multi_pred[ch_idx],dim=0).detach().cpu().numpy()
                all_gt_ch = torch.cat(all_multi_gt[ch_idx], dim=0).detach().cpu().numpy()
                precision, recall,_ = metrics_utils.precision_recall(all_pred_ch, all_gt_ch)
                score = metrics_utils.map_score(all_pred_ch, all_gt_ch)
                fig_pr_curve = unet_utils.plot_precision_recall_curve(precision, recall)

                fig_dict[f'multi_pr_curve_{ch_idx}'] = fig_pr_curve
                metric_dict[f'multi_map_{ch_idx}'] = score
        class_aps = []
        class_ious = []
        mclass_aps = []
        mclass_ious = []
        for k,v in metric_dict.items():
            if 'map_' in k:
                class_aps.append(v)
            elif 'iou_' in k:
                class_ious.append(v)
            if n_classes > 2 and hp['report_multi_class_ap']:
                if 'multi_map_' in k:
                    mclass_aps.append(v)
                elif 'multi_iou_' in k:
                    mclass_ious.append(v)
        metric_dict[f'mean_ap'] = np.mean(class_aps)
        metric_dict[f'mean_iou'] = np.mean(class_ious)
        if n_classes > 2 and hp['report_multi_class_ap']:
            metric_dict[f'multi_mean_ap'] = np.mean(mclass_aps)
            metric_dict[f'multi_mean_iou'] = np.mean(mclass_ious)
    
    if hp['predict_precondition']:
        for ch_idx in all_precond_pred.keys():
            # for plotting precision-recall
            all_pred_ch = np.concatenate(all_precond_pred[ch_idx])
            all_gt_ch = np.concatenate(all_precond_gt[ch_idx])
            precision, recall,thresholds = metrics_utils.precision_recall(all_pred_ch, all_gt_ch)
            metric_dict['pr_auc'] = metrics_utils.get_auc(recall, precision)
            fig_pr_curve = unet_utils.plot_precision_recall_curve(precision, recall)
            fscore = (2 * precision * recall) / (precision + recall)
            # import pdb;pdb.set_trace()
            ix = np.argmax(fscore)
            #print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
            metric_dict['fscore'] = fscore[ix]
            
            roc_auc = metrics_utils.get_roc_auc_score(all_pred_ch, all_gt_ch.astype(bool))
            metric_dict['roc_auc'] = roc_auc
            metric_dict[f'precond_best_thresh_{ch_idx}'] = thresholds[ix]
            fig_dict[f'precond_pr_curve_{ch_idx}'] = fig_pr_curve
   
    return fig_dict, dict(metric_dict)