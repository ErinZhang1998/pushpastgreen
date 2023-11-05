import io 
import os
import PIL
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import utils.plant_utils as plant_utils
import torchvision.transforms as T

def draw_area_want_to_reveal(img):
    
    draw = False
    window_name = "Paint Brush Application"
    color_win_position = [(400, 30), (490,90)]
    bgr_track = {'B': 0, 'G': 0, 'R': 0}

    img2 = np.zeros_like(img)
    cv2.namedWindow(window_name)

    # Initial color window, showing black
    cv2.rectangle(img, color_win_position[0], color_win_position[1], (0,0,0), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, "R: ", (10, 30), font, 0.5, (255,255,255), 1)
    img = cv2.putText(img, "G: ", (90, 30), font, 0.5, (255,255,255), 1)
    img = cv2.putText(img, "B: ", (170, 30), font, 0.5, (255,255,255), 1)

    img = cv2.putText(img, "0", (30, 30), font, 0.5, (255,255,255), 1)
    img = cv2.putText(img, "0", (110, 30), font, 0.5, (255,255,255), 1)
    img = cv2.putText(img, "0", (190, 30), font, 0.5, (255,255,255), 1)

    def nothing(x):
        pass

    def update_R_value(x):
        global font, img, bgr_track
        img = cv2.putText(img, f"{bgr_track['R']}", (30, 30), font, 0.5, (0,0,0), 1)
        img = cv2.putText(img, f"{x}", (30, 30), font, 0.5, (255,255,255), 1)
        bgr_track['R'] = x

    def update_G_value(x):
        global font, img, bgr_track
        img = cv2.putText(img, f"{bgr_track['G']}", (110, 30), font, 0.5, (0,0,0), 1)
        img = cv2.putText(img, f"{x}", (110, 30), font, 0.5, (255,255,255), 1)
        bgr_track['G'] = x

    def update_B_value(x):
        global font, img, bgr_track
        img = cv2.putText(img, f"{bgr_track['B']}", (190, 30), font, 0.5, (0,0,0), 1)
        img = cv2.putText(img, f"{x}", (190, 30), font, 0.5, (255,255,255), 1)
        bgr_track['B'] = x

    def draw_circle(event, x, y, flags, param):
        # global draw, img, img2

        if event == cv2.EVENT_LBUTTONDOWN:
            draw = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if draw:
                cv2.circle(img, (x,y), cv2.getTrackbarPos("Brush Size", window_name),
                        (cv2.getTrackbarPos("B", window_name),
                            cv2.getTrackbarPos("G", window_name),
                            cv2.getTrackbarPos("R", window_name)),
                        -1)
                cv2.circle(img2, (x,y), cv2.getTrackbarPos("Brush Size", window_name),
                        (cv2.getTrackbarPos("B", window_name),
                            cv2.getTrackbarPos("G", window_name),
                            cv2.getTrackbarPos("R", window_name)),
                        -1)

        elif event==cv2.EVENT_LBUTTONUP:
            draw = False
            cv2.circle(img, (x,y), cv2.getTrackbarPos("Brush Size", window_name),
                        (cv2.getTrackbarPos("B", window_name),
                            cv2.getTrackbarPos("G", window_name),
                            cv2.getTrackbarPos("R", window_name)),
                        -1)
            cv2.circle(img2, (x,y), cv2.getTrackbarPos("Brush Size", window_name),
                        (cv2.getTrackbarPos("B", window_name),
                            cv2.getTrackbarPos("G", window_name),
                            cv2.getTrackbarPos("R", window_name)),
                        -1)

    cv2.createTrackbar("R", window_name, 1 ,255, update_R_value)
    cv2.createTrackbar("G", window_name, 1, 255, update_G_value)
    cv2.createTrackbar("B", window_name, 160, 255, update_B_value)
    cv2.createTrackbar("Brush Size", window_name, 5, 15, nothing)
    cv2.setMouseCallback(window_name, draw_circle)

    while(1):
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xff
        if key==ord('q'):
            # cv2.imshow("test", img2)
            # cv2.waitKey(0)
            break

        b = cv2.getTrackbarPos("B", window_name)
        g = cv2.getTrackbarPos("G", window_name)
        r = cv2.getTrackbarPos("R", window_name)
        cv2.rectangle(img, color_win_position[0], color_win_position[1], (b,g,r), -1)

    cv2.destroyAllWindows()
    area_selected = img2[:,:,0] > 0
    return area_selected

def mask_into_black_white(mask, one_is_white=True):
    overlay = mask.astype(np.uint8) * 255
    if not one_is_white:
        overlay = 255-overlay
    overlay_3channel = np.zeros((mask.shape[0], mask.shape[1], 3)).astype(np.uint8)
    overlay_3channel[:,:,0] = overlay
    overlay_3channel[:,:,1] = overlay
    overlay_3channel[:,:,2] = overlay
    return overlay_3channel

class AxIter:
    def __init__(self, axes):
        self.axes = []
        for a in axes:
            
            if type(a) is np.ndarray:
                for b in a:
                    self.axes.append(b)
            else:
                self.axes.append(a)
        self.current = -1
        self.high = len(self.axes)
                

    def __iter__(self):
        return self

    def __next__(self): # Python 2: def next(self)
        self.current += 1
        if self.current < self.high:
            return self.axes[self.current]
        raise StopIteration

def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

def create_axes(num_img_to_visualize, num_pngs_per_row, plot_size=4):
    
    num_rows = num_img_to_visualize // num_pngs_per_row
    if (num_rows * num_pngs_per_row < num_img_to_visualize):
        num_rows += 1
    fig, axes = plt.subplots(num_rows, num_pngs_per_row, figsize=(num_pngs_per_row * plot_size, num_rows * plot_size))
    # plt.margins(0,0)
    axes_iterator = AxIter(axes)
    return fig, axes_iterator

BACKGROUND_COLOR = (1.0, 1.0, 0.6, 1.0)
HEIGHT_DECREASE_COLOR = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0)
HEIGHT_INCREASE_COLOR = (0.984313725490196, 0.7058823529411765, 0.6823529411764706, 1.0)
MASK_COLOR = (0.8, 0.8, 0.8, 1.0)
def create_colormap(num_classes, threshold_classes):
    color_fn = plt.cm.get_cmap('tab20c', 20)
    colormap_colors = []
    for i in range(threshold_classes):
        colormap_colors.append(color_fn(7-i))
    for i in range(num_classes-threshold_classes):
        if i < 4:
            colormap_colors.append(color_fn(3-i))
        else:
            colormap_colors.append(color_fn(12+i))
    colormap_colors.append(MASK_COLOR)
    return  colormap_colors

# colormap_colors = [BACKGROUND_COLOR,HEIGHT_DECREASE_COLOR,(0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0),(0.6509803921568628, 0.8470588235294118, 0.32941176470588235, 1.0),MASK_COLOR]
# mask_color_map = mpl.colors.LinearSegmentedColormap.from_list("", colormap_colors)

def plot_postactionmask(mask_gt, mask_pred_softmax, action, ax_iter, title = None):
    ax = next(ax_iter)
    ax.imshow(mask_gt)
    ax.arrow(action[0], action[1], action[2]-action[0], action[3]-action[1], length_includes_head=True,head_width=4,width=2, head_length=5,color='r')
    ax.axis("off")

    ax = next(ax_iter)
    im = ax.imshow(mask_pred_softmax, vmin=0, vmax=1, cmap=mpl.colormaps['YlGn'])
    ax.arrow(action[0], action[1], action[2]-action[0], action[3]-action[1], length_includes_head=True,head_width=4,width=2, head_length=5,color='r')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.axis("off")
    plt.colorbar(im, cax=cax)

def plot_plant_img_and_mask(img, img_mask, height_img, mask_gt, mask_pred_argmax, mask_pred_softmax, height_diff_img, height_diff_pred, action, ax_iter, colormap_colors, title = None, num_classes = 4):
    mask_color_map = mpl.colors.LinearSegmentedColormap.from_list("", colormap_colors)
    ax = next(ax_iter)
    im = ax.imshow(height_img,vmin=0, vmax=1) # vmin=0, vmax=1, cmap=mpl.colormaps['bwr'] ,cmap=mpl.colormaps['YlGn']
    ax.arrow(action[0], action[1], action[2]-action[0], action[3]-action[1], length_includes_head=True,head_width=17,width=3, head_length=10,color='r')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.axis("off")
    plt.colorbar(im, cax=cax)

    ax = next(ax_iter)
    ax.imshow(img) # vmin=0, vmax=1, cmap=mpl.colormaps['bwr']
    ax.arrow(action[0], action[1], action[2]-action[0], action[3]-action[1], length_includes_head=True,head_width=17,width=3, head_length=10,color='r')
    ax.axis("off")

    ax = next(ax_iter)
    # mask_gt[np.where(img_mask)] = len(colormap_colors)-1
    # ax.imshow(mask_gt, cmap=mask_color_map)
    mask_gt[np.where(img_mask)] = -1
    im = ax.imshow(mask_gt, vmin = -1, vmax = num_classes-1)
    ax.arrow(action[0], action[1], action[2]-action[0], action[3]-action[1], length_includes_head=True,head_width=17,width=3, head_length=10,color='r')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.axis("off")
    plt.colorbar(im, cax=cax)

    ax = next(ax_iter)
    mask_pred_argmax[0][0] = 0
    # mask_pred_argmax[np.where(img_mask)] = len(colormap_colors)-1
    # ax.imshow(mask_pred_argmax, cmap=mask_color_map)
    mask_pred_argmax[np.where(img_mask)] = -1
    im = ax.imshow(mask_pred_argmax, vmin = -1, vmax = num_classes-1)
    ax.arrow(action[0], action[1], action[2]-action[0], action[3]-action[1], length_includes_head=True,head_width=17,width=3, head_length=10,color='r')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.axis("off")
    plt.colorbar(im, cax=cax)
    if title:
        ax.set_title(title, fontdict = {'fontsize': 8})

    ax = next(ax_iter)
    im = ax.imshow(mask_pred_softmax, vmin=0, vmax=1)
    ax.arrow(action[0], action[1], action[2]-action[0], action[3]-action[1], length_includes_head=True,head_width=17,width=3, head_length=10,color='r')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.axis("off")
    plt.colorbar(im, cax=cax)

    if height_diff_img is not None:
        ax = next(ax_iter)
        im = ax.imshow(height_diff_img, vmin=0, vmax=1)
        ax.arrow(action[0], action[1], action[2]-action[0], action[3]-action[1], length_includes_head=True,head_width=17,width=3, head_length=10,color='r')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.axis("off")
        plt.colorbar(im, cax=cax)
    if height_diff_pred is not None:
        ax = next(ax_iter)
        im = ax.imshow(height_diff_pred, vmin=0, vmax=1)
        ax.arrow(action[0], action[1], action[2]-action[0], action[3]-action[1], length_includes_head=True,head_width=17,width=3, head_length=10,color='r')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.axis("off")
        plt.colorbar(im, cax=cax)

def plt_to_image(fig_obj):
    buf = io.BytesIO()
    fig_obj.savefig(buf)
    buf.seek(0)
    return PIL.Image.open(buf)

def bound_pixel(coord, size):
    return min(max(0, coord), size-1)
    if coord < 0:
        return 0 
    if coord >= size:
        return size-1
    return coord

def flip_end_pixel(start_pixel, end_pixel, size):
    dx = end_pixel[0] - start_pixel[0]
    new_end_pixel = [bound_pixel(start_pixel[0] + (-dx), size), end_pixel[1]]
    return new_end_pixel

def plot_precision_recall_curve(precision, recall):
    #create precision recall curve
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    
    ax.scatter(recall, precision, color='purple', s=5)
    ax.plot(recall, precision, color='purple', alpha=0.5)

    #add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    #display plot
    return fig

IMG_SIZE = 150



nov_11_angle_idx_to_angle_list = {0: 0, 1:30, 2:60, 3:90,4:120,5:150,6:180}
nov_11_angle_to_angle_idx_list = {0: 0, 30:1, 60:2, 90:3,120:4,150:5,180:6}

def visualize_space_revealed_by_actions(
    initial_space_revealed_map, 
    x_start_calculate, x_end_calculate, y_start_calculate, y_end_calculate, 
    x_start_action, x_end_action, y_start_action, y_end_action, 
    this_action_info, score, fig_save_path = None, num_figures_per_row = 5,
    is_forward_model = True):
    # import pickle
    # this_action_info, score = pickle.load(open(this_action_info_saved_path, "rb"))
    nactions = len(this_action_info)
    fig, axes_iterator = create_axes(nactions * num_figures_per_row+num_figures_per_row, num_figures_per_row, plot_size=6)
    ax = next(axes_iterator)
    h,w = initial_space_revealed_map.shape
    initial_space = 255 - initial_space_revealed_map.astype(np.uint8) * 255
    initial_space_revealed_map_3channel = np.zeros((h,w,3)).astype(np.uint8)
    initial_space_revealed_map_3channel[:,:,0] = initial_space
    initial_space_revealed_map_3channel[:,:,1] = initial_space
    initial_space_revealed_map_3channel[:,:,2] = initial_space

    ax.imshow(initial_space_revealed_map_3channel)
    rect = patches.Rectangle((x_start_calculate, y_start_calculate), x_end_calculate-x_start_calculate+1, y_end_calculate-y_start_calculate+1, linewidth=1, edgecolor='pink', facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle((x_start_action, y_start_action), x_end_action-x_start_action+1, y_end_action-y_start_action+1, linewidth=1, edgecolor='lightgreen', facecolor='none')
    
    for step_idx, info in enumerate(this_action_info):
        action = info["reveal_map_action_pixels"]
        ax.arrow(action[0], action[1], action[2]-action[0], action[3]-action[1], length_includes_head=True,head_width=17,width=3, head_length=10,color='r')
    
    ax.add_patch(rect)
    title = f"score={score:.2f}"
    ax.set_title(title, fontdict = {'fontsize': 8})
   
   
    for _ in range(num_figures_per_row-1):
        ax = next(axes_iterator)
        ax.remove()
    
    
    for step_idx, info in enumerate(this_action_info):
        # x,y,theta,length  = info["action"]
        # angle = np.degrees(theta)
        current_space_revealed_map = info["current_space_revealed_map"].copy()
        
        background = initial_space_revealed_map_3channel
        overlay = 255 - current_space_revealed_map.astype(np.uint8) * 255
        overlay_3channel = np.zeros_like(background)
        overlay_3channel[:,:,0] = overlay
        overlay_3channel[:,:,1] = overlay
        overlay_3channel[:,:,2] = overlay
        new_space_revealed_on_space_revealed_map_before_action = cv2.addWeighted(background,0.2,overlay_3channel,0.7,0)

        ax = next(axes_iterator)
        ax.imshow(new_space_revealed_on_space_revealed_map_before_action)
        
        action = info["reveal_map_action_pixels"]
        ax.arrow(action[0], action[1], action[2]-action[0], action[3]-action[1], length_includes_head=True,head_width=17,width=3, head_length=10,color='r')
        rect = patches.Rectangle((x_start_calculate, y_start_calculate), x_end_calculate-x_start_calculate+1, y_end_calculate-y_start_calculate+1, linewidth=1, edgecolor='pink', facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((x_start_action, y_start_action), x_end_action-x_start_action+1, y_end_action-y_start_action+1, linewidth=1, edgecolor='lightgreen', facecolor='none')
        ax.add_patch(rect)
        if  is_forward_model:
            rect = patches.Rectangle((info["reveal_map_x_start"], info["reveal_map_y_start"]), info["pred"].shape[0], info["pred"].shape[1], linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        if step_idx == len(this_action_info) - 1:
            new_discovery = info["current_space_revealed_map"] - initial_space_revealed_map
            new_discovery = new_discovery[y_start_calculate:y_end_calculate+1, x_start_calculate:x_end_calculate+1]
            re_score = new_discovery[new_discovery > 0].sum()
            ax.set_title(f"{re_score:.2f}", fontdict = {'fontsize': 8})

        if not is_forward_model:
            continue 

        if 'action_start_pixels' in info and 'action_end_pixels' in info:
            action_in_original = [
                info["action_start_pixels"][0],
                info["action_start_pixels"][1],
                info["action_end_pixels"][0],
                info["action_end_pixels"][1],
            ]
            add_arrow = True
            def add_arow_fn(ax):
                ax.arrow(action_in_original[0], action_in_original[1], action_in_original[2]-action_in_original[0], action_in_original[3]-action_in_original[1], length_includes_head=True,head_width=17,width=3, head_length=10,color='r')
        else:
            add_arrow = False
            def add_arow_fn(ax):
                return
        
        ax = next(axes_iterator)
        im = ax.imshow(info["height_image_input"],vmin=0, vmax=1) # vmin=0, vmax=1, cmap=mpl.colormaps['bwr']
        add_arow_fn(ax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.axis("off")
        plt.colorbar(im, cax=cax)
        # title = f"{x:.2f} {y:.2f} {angle:.2f} {length:.2f}"
        # ax.set_title(title, fontdict = {'fontsize': 8})

        ax = next(axes_iterator)
        ax.imshow(info["rgb_image_input"]) # vmin=0, vmax=1, cmap=mpl.colormaps['bwr']
        add_arow_fn(ax)
        ax.axis("off")
        title = f"{action_in_original[0]:.1f} {action_in_original[1]:.1f} {action_in_original[2]:.1f} {action_in_original[3]:.1f}"
        ax.set_title(title, fontdict = {'fontsize': 8})

        
        ax = next(axes_iterator)
        ax.imshow(info["pred"])
        add_arow_fn(ax)
        
        ax = next(axes_iterator)
        ax.imshow(info['pred_gt'])
        add_arow_fn(ax)

        ax = next(axes_iterator)
        ax.imshow(info["pred_prob"], vmin=0, vmax=1, cmap=mpl.colormaps['YlGn'])
        add_arow_fn(ax)

    if fig_save_path is not None:
        final_img = plt_to_image(fig)
        final_img.save(fig_save_path) 
    return fig


def visualize_best_action(pi_t, local_patch_size = 150, is_forward_model = True):
    best_action_sequence = pi_t['best_actions_processed'][0] # (nactions, 5)
    config = pi_t['policy_param']
    img_input_resolution = config['img_input_resolution']
    workspace_for_reveal = np.asarray(config["workspace_for_reveal"])
    space_revealed_map_h, space_revealed_map_w = config["space_revealed_map_h"], config["space_revealed_map_w"]
    x_start_calculate, x_end_calculate, y_start_calculate, y_end_calculate = \
        np.asarray(config["calculate_bounds_in_space_revealed_map"]).astype(int)
    x_start_action, x_end_action, y_start_action, y_end_action = \
        np.asarray(config["action_bounds_in_space_revealed_map"]).astype(int)
    infos = []
    for step_idx, actions_processed in enumerate(best_action_sequence):
        pt_x,pt_y,pt_z,theta,length = actions_processed.reshape(-1,)[:5]
        
        dx = np.cos(theta) * length
        dy = np.sin(theta) * length
        start_x_pixels, start_y_pixels = plant_utils.get_coordinate_in_image(np.array([[pt_x,pt_y], [pt_x+dx,pt_y+dy]]), workspace_for_reveal[0][0], workspace_for_reveal[1][0], space_revealed_map_w, space_revealed_map_h, img_input_resolution, restrict_to_within_image=True)
        x_start = start_x_pixels[0]
        y_start = start_y_pixels[0]
        reveal_map_action_pixels = [x_start, y_start, start_x_pixels[1], start_y_pixels[1]]
        current_space_revealed_map = pi_t['current_space_revealed_map'][step_idx]

        info = {
            "pt_x" : pt_x, 
            "pt_y" : pt_y, 
            'reveal_map_action_pixels' : reveal_map_action_pixels, # in the whole image of keeping track of space revealed 
            "current_space_revealed_map" : current_space_revealed_map, 
        }
        if is_forward_model:
            x_end = max(
                min(local_patch_size//2 + dx / img_input_resolution, local_patch_size-1), 0)
            y_end = max(
                min(local_patch_size//2 - dy / img_input_resolution, local_patch_size-1), 0)

            xpixel, ypixel = pi_t['reveal_map_xy_start'][step_idx].astype(int)
            
            info.update({
                'reveal_map_x_start' : xpixel, # where is this patch in the reveal map
                'reveal_map_y_start' : ypixel,
                "action_start_pixels": [local_patch_size//2,local_patch_size//2-1], # where in the patch does the action start
                "action_end_pixels" : [int(x_end), int(y_end)], 
                "pred" : pi_t['pred'][step_idx], 
                "pred_prob" : pi_t['pred_prob'][step_idx], 
                "rgb_image_input" : T.ToPILImage()(pi_t['rgb_image_input'][step_idx]), 
                "height_image_input" : pi_t['height_image_input'][step_idx],
            })
            if "new_space_revealed_map_gt" in pi_t:
                pred_gt = pi_t["new_space_revealed_map_gt"][ypixel:ypixel+info['pred'].shape[0], xpixel:xpixel+info['pred'].shape[1]]
            else:
                pred_gt = np.zeros_like(pi_t['pred'][step_idx])
            info['pred_gt'] = pred_gt
        infos.append(info)

    fig = visualize_space_revealed_by_actions(
        np.copy(pi_t["space_revealed_map_before_action"]), \
        x_start_calculate, x_end_calculate, \
        y_start_calculate, y_end_calculate, \
        x_start_action, x_end_action, \
        y_start_action, y_end_action, \
        infos, \
        0.0,
        num_figures_per_row = 2 if not is_forward_model else 6,
        is_forward_model = is_forward_model,
    )
    return fig

def visualize_cem(t, policy_output, workspace_for_reveal, space_revealed_map_w, space_revealed_map_h, img_input_resolution, calculate_bound, action_bound, visualize_selected_theta = False, visualize_not_selected_theta=False, only_last_itr=False):
    x_start_calculate, x_end_calculate, y_start_calculate, y_end_calculate = calculate_bound
    x_start_action, x_end_action, y_start_action, y_end_action = action_bound

    import cv2
    
    color_name_dict = {
        0 : "blue",
        1 : "yellow",
        2 : "green",
        3 : "red",
        4 : "olive",
        5 : "grey",
        6 : "maroon",
    }
    import matplotlib as mpl
    total_itr = int(policy_output['plan_stat']['itr'])
    total_itr += 1
    space_revealed_map_before_action = policy_output["space_revealed_map_before_action"]
    background = policy_output['start_heightmaps'][0]
    overlay = 255 - space_revealed_map_before_action.astype(np.uint8) * 255
    overlay_3channel = np.zeros_like(background)
    overlay_3channel[:,:,0] = overlay
    overlay_3channel[:,:,1] = overlay
    overlay_3channel[:,:,2] = overlay
    space_revealed_map_before_action_on_color_heightmap = cv2.addWeighted(background,0.7,overlay_3channel,0.5,0)

    if only_last_itr:
        fig, axes_iterator = create_axes(2, 2, plot_size=6)
    else:
        fig, axes_iterator = create_axes(total_itr+1, 2, plot_size=12)
    ax = next(axes_iterator)
    ax.imshow(space_revealed_map_before_action_on_color_heightmap)
    rect = patches.Rectangle((x_start_action, y_start_action), \
                x_end_action-x_start_action+1, y_end_action-y_start_action+1, \
                            linewidth=1, edgecolor='lightgreen', facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle((x_start_calculate, y_start_calculate), \
        x_end_calculate-x_start_calculate+1, y_end_calculate-y_start_calculate+1, \
                        linewidth=1, edgecolor='pink', facecolor='none')
    ax.add_patch(rect)

    step_t = 0
    for itr in range(total_itr):
        if only_last_itr and itr < total_itr-1:
            continue
        ax = next(axes_iterator)
        ax.set_title(f"{t}_{itr}")
        processed_actions_not_selected = policy_output['plan_stat'][f'processed_actions_not_selected_{itr}']
        processed_actions_best = policy_output['plan_stat'][f'processed_actions_best_{itr}']
        
        x_coords, y_coords = plant_utils.get_coordinate_in_image(processed_actions_not_selected[:,step_t,:2], workspace_for_reveal[0][0], workspace_for_reveal[1][0], space_revealed_map_w, space_revealed_map_h, img_input_resolution, restrict_to_within_image=True)
        im = ax.imshow(space_revealed_map_before_action_on_color_heightmap, alpha=0.8)
        rect = patches.Rectangle((x_start_action, y_start_action), \
                    x_end_action-x_start_action+1, y_end_action-y_start_action+1, \
                                linewidth=1, edgecolor='lightgreen', facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((x_start_calculate, y_start_calculate), \
            x_end_calculate-x_start_calculate+1, y_end_calculate-y_start_calculate+1, \
                         linewidth=1, edgecolor='pink', facecolor='none')
        ax.add_patch(rect)
        ax.scatter(x_coords, y_coords, s=10, color="red")
        if visualize_not_selected_theta:
            thetas = processed_actions_not_selected[:,step_t,3]
            dxs = np.cos(thetas) * processed_actions_not_selected[:,step_t,4]
            dys = np.sin(thetas) * processed_actions_not_selected[:,step_t,4]
            dxs /= img_input_resolution
            dys /= img_input_resolution
            dxs = dxs.astype(int)
            dys = dys.astype(int)
            angles = np.floor((thetas - 0.0) / (np.pi / 6)).astype(int)
        
            for idx,(x,y) in enumerate(zip(x_coords,y_coords)):
                if angles[idx] < 4:
                    continue
                color = color_name_dict[angles[idx]]
                ax.arrow(x, y, dxs[idx], -dys[idx], \
                        length_includes_head=True,head_width=4,width=2, head_length=5,color=color, alpha=0.3)
            
        x_coords, y_coords = plant_utils.get_coordinate_in_image(processed_actions_best[:,0,:2], \
                workspace_for_reveal[0][0], workspace_for_reveal[1][0], \
                space_revealed_map_w, space_revealed_map_h, \
                                            img_input_resolution, restrict_to_within_image=True)
        ax.scatter(x_coords, y_coords, s=10, color="blue")
        
        if visualize_selected_theta:
            thetas = processed_actions_best[:,step_t,3]
            dxs = np.cos(thetas) * processed_actions_best[:,step_t,4]
            dys = np.sin(thetas) * processed_actions_best[:,step_t,4]
            dxs /= img_input_resolution
            dys /= img_input_resolution
            dxs = dxs.astype(int)
            dys = dys.astype(int)
            angles = np.floor((thetas - 0.0) / (np.pi / 6)).astype(int)
            
            for idx,(x,y) in enumerate(zip(x_coords,y_coords)):
                if angles[idx] < 4:
                    continue
                color = color_name_dict[angles[idx]]
                ax.arrow(x, y, \
                        dxs[idx], -dys[idx], \
                        length_includes_head=True,head_width=4,width=2, head_length=5,color=color, alpha=0.5)
            

    return fig

def visualize_cem_no_overlay(policy_output, workspace_for_reveal, space_revealed_map_w, space_revealed_map_h, img_input_resolution, calculate_bound, action_bound, visualize_selected_theta = False, visualize_not_selected_theta=False, only_last_itr=False):
    x_start_calculate, x_end_calculate, y_start_calculate, y_end_calculate = calculate_bound
    x_start_action, x_end_action, y_start_action, y_end_action = action_bound

    import cv2

    import matplotlib as mpl
    total_itr = int(policy_output['plan_stat']['itr'])
    total_itr += 1
    space_revealed_map_before_action = policy_output["space_revealed_map_before_action"]
    background = policy_output['start_heightmaps'][0]
    overlay = space_revealed_map_before_action.astype(np.uint8) * 255
    overlay_3channel = np.zeros_like(background)
    overlay_3channel[:,:,0] = overlay
    overlay_3channel[:,:,1] = overlay
    overlay_3channel[:,:,2] = overlay
    space_revealed_map_before_action_on_color_heightmap = cv2.addWeighted(background,0.7,overlay_3channel,0.5,0)
    
    fig,ax = plt.subplots(figsize=(10,10))
    ax.imshow(overlay_3channel[y_start_calculate:y_end_calculate+1, x_start_calculate:x_end_calculate+1])
    ax.axis("off")
    plt.show()

    step_t = 0
    for itr in range(total_itr):
        if only_last_itr and itr < total_itr-1:
            continue
        fig,ax = plt.subplots(figsize=(10,10))
        processed_actions_not_selected = policy_output['plan_stat'][f'processed_actions_not_selected_{itr}']
        processed_actions_best = policy_output['plan_stat'][f'processed_actions_best_{itr}']
        
        x_coords, y_coords = plant_utils.get_coordinate_in_image(processed_actions_not_selected[:,step_t,:2], workspace_for_reveal[0][0], workspace_for_reveal[1][0], space_revealed_map_w, space_revealed_map_h, img_input_resolution, restrict_to_within_image=True)
        x_coords -= x_start_calculate
        y_coords -= y_start_calculate
        im = ax.imshow(background[y_start_calculate:y_end_calculate+1, x_start_calculate:x_end_calculate+1], alpha=0.8)
        
        ax.scatter(x_coords, y_coords, s=30, color="red")
        x_coords, y_coords = plant_utils.get_coordinate_in_image(processed_actions_best[:,0,:2], \
                workspace_for_reveal[0][0], workspace_for_reveal[1][0], \
                space_revealed_map_w, space_revealed_map_h, \
                                            img_input_resolution, restrict_to_within_image=True)
        x_coords -= x_start_calculate
        y_coords -= y_start_calculate
        ax.scatter(x_coords, y_coords, s=30, color="blue")
        ax.axis("off")
        plt.show()

def calculate_num_layers_to_min_dimension(input_dimension, min_dimension):
    size = input_dimension
    num_layers = 0
    while size > min_dimension:
        num_layers += 1
        size = size // 2
    return num_layers,size

def calculate_num_layers_to_max_dimension(input_dimension, max_dimension):
    size = input_dimension
    num_layers = 0
    while True:
        num_layers += 1
        size = size * 2
        if size * 2 > max_dimension:
            break
    return num_layers,size

def visualize_best_action(pi_t, local_patch_size = 150, is_forward_model = True):
    best_action_sequence = pi_t['best_actions_processed'][0] # (nactions, 5)
    config = pi_t['policy_param']
    img_input_resolution = config['img_input_resolution']
    workspace_for_reveal = np.asarray(config["workspace_for_reveal"])
    space_revealed_map_h, space_revealed_map_w = config["space_revealed_map_h"], config["space_revealed_map_w"]
    x_start_calculate, x_end_calculate, y_start_calculate, y_end_calculate = \
        np.asarray(config["calculate_bounds_in_space_revealed_map"]).astype(int)
    x_start_action, x_end_action, y_start_action, y_end_action = \
        np.asarray(config["action_bounds_in_space_revealed_map"]).astype(int)
    infos = []
    for step_idx, actions_processed in enumerate(best_action_sequence):
        pt_x,pt_y,pt_z,theta,length = actions_processed.reshape(-1,)
        
        dx = np.cos(theta) * length
        dy = np.sin(theta) * length
        start_x_pixels, start_y_pixels = plant_utils.get_coordinate_in_image(np.array([[pt_x,pt_y], [pt_x+dx,pt_y+dy]]), workspace_for_reveal[0][0], workspace_for_reveal[1][0], space_revealed_map_w, space_revealed_map_h, img_input_resolution, restrict_to_within_image=True)
        x_start = start_x_pixels[0]
        y_start = start_y_pixels[0]
        reveal_map_action_pixels = [x_start, y_start, start_x_pixels[1], start_y_pixels[1]]
        current_space_revealed_map = pi_t['current_space_revealed_map'][step_idx]

        info = {
            "pt_x" : pt_x, 
            "pt_y" : pt_y, 
            'reveal_map_action_pixels' : reveal_map_action_pixels,
            "current_space_revealed_map" : current_space_revealed_map, 
        }
        if is_forward_model:
            x_end = max(
                min(local_patch_size//2 + dx / img_input_resolution, local_patch_size-1), 0)
            y_end = max(
                min(local_patch_size//2 - dy / img_input_resolution, local_patch_size-1), 0)

            xpixel, ypixel = pi_t['reveal_map_xy_start'][step_idx].astype(int)
            info.update({
                'reveal_map_x_start' : xpixel,
                'reveal_map_y_start' : ypixel,
                "action_start_pixels": [local_patch_size//2,local_patch_size//2-1], 
                "action_end_pixels" : [int(x_end), int(y_end)], 
                "pred" : pi_t['pred'][step_idx], 
                "pred_prob" : pi_t['pred_prob'][step_idx], 
                "rgb_image_input" : T.ToPILImage()(pi_t['rgb_image_input'][step_idx]), 
                "height_image_input" : pi_t['height_image_input'][step_idx],
            })
        
        infos.append(info)

    fig = visualize_space_revealed_by_actions(
        np.copy(pi_t["space_revealed_map_before_action"]), \
        x_start_calculate, x_end_calculate, \
        y_start_calculate, y_end_calculate, \
        x_start_action, x_end_action, \
        y_start_action, y_end_action, \
        infos, \
        0.0,
        num_figures_per_row = 2 if not is_forward_model else 5,
        is_forward_model = is_forward_model,
    )
    return fig

def visualize_forward_model_outputs(info):
    fig, axes_iterator = create_axes(1, 4, plot_size=6)
    # xpixel, ypixel = info["reveal_map_x_start"], info['reveal_map_y_start']
    if 'action_start_pixels' in info and 'action_end_pixels' in info:
        action_in_original = [
            info["action_start_pixels"][0],
            info["action_start_pixels"][1],
            info["action_end_pixels"][0],
            info["action_end_pixels"][1],
        ]
        add_arrow = True
    else:
        add_arrow = False
        
    
    ax.imshow(info["rgb_image_input"]) # vmin=0, vmax=1, cmap=mpl.colormaps['bwr']
    if add_arrow:
        ax.arrow(action_in_original[0], action_in_original[1], action_in_original[2]-action_in_original[0], action_in_original[3]-action_in_original[1], length_includes_head=True,head_width=15,width=3, head_length=10,color='r')
    ax.axis("off")
    
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(info["height_image_input"],vmin=0, vmax=1,cmap=mpl.colormaps['YlGn']) # vmin=0, vmax=1, cmap=mpl.colormaps
    if add_arrow:
        ax.arrow(action_in_original[0], action_in_original[1], action_in_original[2]-action_in_original[0], action_in_original[3]-action_in_original[1], length_includes_head=True,head_width=15,width=3, head_length=10,color='r')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.axis("off")
    # plt.colorbar(im, cax=cax)

    ax = next(axes_iterator)
    gt_mask = info['gt'] #[ypixel:ypixel+info['pred'].shape[0], xpixel:xpixel+info['pred'].shape[1]]
    gt_mask_3_channel = mask_into_black_white(gt_mask, one_is_white=True)
    ax.imshow(gt_mask_3_channel)
    # ax.imshow(info["pred_prob"], vmin=0, vmax=1, cmap=mpl.colormaps['YlGn'])
    if add_arrow:
        ax.arrow(action_in_original[0], action_in_original[1], action_in_original[2]-action_in_original[0], action_in_original[3]-action_in_original[1], length_includes_head=True,head_width=15,width=3, head_length=10,color='r')
    ax.axis("off")

    ax = next(axes_iterator)
    pred_3_channel = mask_into_black_white(info["pred"], one_is_white=True)
    ax.imshow(pred_3_channel)
    if add_arrow:
        ax.arrow(action_in_original[0], action_in_original[1], action_in_original[2]-action_in_original[0], action_in_original[3]-action_in_original[1], length_includes_head=True,head_width=15,width=3, head_length=10,color='r')
    ax.axis("off")
    plt.margins(0,0)


def visualize_forward_model_outputs_save_separately(info):
    if 'action_start_pixels' in info and 'action_end_pixels' in info:
        action_in_original = [
            info["action_start_pixels"][0],
            info["action_start_pixels"][1],
            info["action_end_pixels"][0],
            info["action_end_pixels"][1],
        ]
        add_arrow = True
    else:
        add_arrow = False
        
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(info["rgb_image_input"]) # vmin=0, vmax=1, cmap=mpl.colormaps['bwr']
    if add_arrow:
        ax.arrow(action_in_original[0], action_in_original[1], action_in_original[2]-action_in_original[0], action_in_original[3]-action_in_original[1], length_includes_head=True,head_width=15,width=3, head_length=10,color='r')
    ax.axis("off")
    path = os.path.join(info['save_folder'], info['save_path_prefix']+"_rgb.pdf")
    plt.savefig(path,  bbox_inches = 'tight', pad_inches = 0.01)
    
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(info["height_image_input"],vmin=0.5, vmax=1,cmap=mpl.colormaps['YlGn']) # vmin=0, vmax=1, cmap=mpl.colormaps
    if add_arrow:
        ax.arrow(action_in_original[0], action_in_original[1], action_in_original[2]-action_in_original[0], action_in_original[3]-action_in_original[1], length_includes_head=True,head_width=15,width=3, head_length=10,color='r')
    ax.axis("off")
    path = os.path.join(info['save_folder'], info['save_path_prefix']+"_height.pdf")
    plt.savefig(path,  bbox_inches = 'tight', pad_inches = 0.01)
    

    fig, ax = plt.subplots(figsize=(5,5))
    gt_mask = info['gt'] #[ypixel:ypixel+info['pred'].shape[0], xpixel:xpixel+info['pred'].shape[1]]
    gt_mask_3_channel = mask_into_black_white(gt_mask, one_is_white=True)
    ax.imshow(gt_mask_3_channel)
    # ax.imshow(info["pred_prob"], vmin=0, vmax=1, cmap=mpl.colormaps['YlGn'])
    if add_arrow:
        ax.arrow(action_in_original[0], action_in_original[1], action_in_original[2]-action_in_original[0], action_in_original[3]-action_in_original[1], length_includes_head=True,head_width=15,width=3, head_length=10,color='r')
    ax.axis("off")
    path = os.path.join(info['save_folder'], info['save_path_prefix']+"_gt.pdf")
    plt.savefig(path,  bbox_inches = 'tight', pad_inches = 0.01)
    

    fig, ax = plt.subplots(figsize=(5,5))
    # pred_3_channel = mask_into_black_white(info["pred"], one_is_white=True)
    # ax.imshow(pred_3_channel)
    pred_prob = 1-info['prob']
    im = ax.imshow(pred_prob, vmin=0, vmax=1, cmap=mpl.colormaps['Greys'])
    if add_arrow:
        ax.arrow(action_in_original[0], action_in_original[1], action_in_original[2]-action_in_original[0], action_in_original[3]-action_in_original[1], length_includes_head=True,head_width=15,width=3, head_length=10,color='r')
    ax.axis("off")
    # path = os.path.join(info['save_folder'], info['save_path_prefix']+"_pred.pdf")
    path = os.path.join(info['save_folder'], info['save_path_prefix']+"_prob.pdf")
    plt.savefig(path,  bbox_inches = 'tight', pad_inches = 0.01)
    
    plt.close("all")
