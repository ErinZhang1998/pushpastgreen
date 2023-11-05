import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import utils.plant_utils as plant_utils
from autolab_core import YamlConfig 
import collections
from scipy.stats import bootstrap
import statsmodels.stats.api as sms
import scipy.stats as st


single_trial_full_folders = [
    ("/scratch0/zhang401local/plant_data/experiments/Jan-31-2023",[
        (0, "Jan-31-2023-12-58-52"),
        (0, "Jan-31-2023-13-11-37"),
        (0, "Jan-31-2023-15-45-17"),
        (0, "Jan-31-2023-17-25-09"),
        (0, "Jan-31-2023-17-26-22"),
        (0, "Jan-31-2023-17-27-31"),
        (0, "Jan-31-2023-17-29-10"),
        (0, "Jan-31-2023-17-30-19"),
        (0, "Jan-31-2023-17-31-20"),
        (0, "Jan-31-2023-17-32-29"),
        (0, "Jan-31-2023-17-33-23"),
        (0, "Jan-31-2023-17-34-23"),
        (0, "Jan-31-2023-17-35-36"),
        (0, "Jan-31-2023-17-36-59"),
        (0, "Jan-31-2023-17-38-01"),
        (0, "Jan-31-2023-17-39-28"),
        (0, "Jan-31-2023-17-40-52"),
        (0, "Jan-31-2023-17-44-50"),
        (0, "Jan-31-2023-17-45-57"),
        (0, "Jan-31-2023-17-47-12"),
        (0, "Jan-31-2023-17-48-11"),
        (0, "Jan-31-2023-17-49-44"),
        (0, "Jan-31-2023-17-51-10"),
        (0, "Jan-31-2023-17-52-12"),
        (0, "Jan-31-2023-17-53-14"),
    ]),
    ("/scratch0/zhang401local/plant_data/experiments/Feb-01-2023",[
        (3, "Feb-01-2023-08-43-05"),
        (3, "Feb-01-2023-08-57-53"),
        (3, "Feb-01-2023-09-11-32"),
        (3, "Feb-01-2023-09-34-19"),
        (3, "Feb-01-2023-09-35-30"),
        (3, "Feb-01-2023-09-37-11"),
        (3, "Feb-01-2023-09-38-30"),
        (3, "Feb-01-2023-09-41-20"),
        (3, "Feb-01-2023-09-42-23"),
        (3, "Feb-01-2023-09-43-15"),
        (3, "Feb-01-2023-09-45-18"),
        (3, "Feb-01-2023-09-46-38"),
        (3, "Feb-01-2023-09-47-30"),
        (3, "Feb-01-2023-09-49-25"),
        (0, "Feb-01-2023-09-51-30"),
        (3, "Feb-01-2023-09-54-48"),
        (3, "Feb-01-2023-09-55-44"),
        (3, "Feb-01-2023-10-35-57"),
        (3, "Feb-01-2023-10-36-50"),
        (3, "Feb-01-2023-10-38-01"),
        (0, "Feb-01-2023-10-39-02"),
        (0, "Feb-01-2023-10-39-47"),
        (3, "Feb-01-2023-10-41-45"),
        (0, "Feb-01-2023-10-42-38"),
        (0, "Feb-01-2023-10-43-23"),
        (3, "Feb-01-2023-10-44-13"),
        (3, "Feb-01-2023-10-46-10"),
        (3, "Feb-01-2023-12-02-54"),
        (3, "Feb-01-2023-12-07-25"),
        (3, "Feb-01-2023-12-08-40"),
        (0, "Feb-01-2023-12-11-02"),
        (3, "Feb-01-2023-12-12-35"),
        (3, "Feb-01-2023-12-15-50"),
        (3, "Feb-01-2023-12-39-06"),
        (3, "Feb-01-2023-12-40-09"),
        (0, "Feb-01-2023-14-28-43"),
        (3, "Feb-01-2023-13-48-46"),
        (3, "Feb-01-2023-14-14-45"),
        (0, "Feb-01-2023-14-30-38"),
        (0, "Feb-01-2023-14-43-52"),
        (3, "Feb-01-2023-15-11-00"),
        (3, "Feb-01-2023-15-18-14"),
        (3, "Feb-01-2023-15-24-59"),
        (3, "Feb-01-2023-15-33-39"),
        (0, "Feb-01-2023-16-08-28"),
    ]),
]

full_folders = [
    ("/scratch0/zhang401local/plant_data/experiments/Jan-31-2023",[ 
        (0, "Jan-31-2023-12-58-52"),
        (0, "Jan-31-2023-13-11-37"),
        (1, "Jan-31-2023-13-25-11"),
        (1, "Jan-31-2023-13-33-40"),
        (1, "Jan-31-2023-14-20-51"),
        (2, "Jan-31-2023-14-28-23"),
        (2, "Jan-31-2023-15-01-06"),
        (2, "Jan-31-2023-15-20-07"),
    ]),
    ("/scratch0/zhang401local/plant_data/experiments/Feb-01-2023",[
        (3, "Feb-01-2023-08-43-05"),
        (3, "Feb-01-2023-08-57-53"),
        (3, "Feb-01-2023-09-11-32"),
        (3, "Feb-01-2023-13-48-46"),
        (1, "Feb-01-2023-13-57-08"),
        (2, "Feb-01-2023-14-06-46"),
        (3, "Feb-01-2023-14-14-45"),
        (1, "Feb-01-2023-14-22-27"),
        (0, "Feb-01-2023-14-30-38"),
        (2, "Feb-01-2023-14-37-20"),
        (0, "Feb-01-2023-14-43-52"),
        (1, "Feb-01-2023-14-50-21"),
        (1, "Feb-01-2023-14-58-03"),
        (1, "Feb-01-2023-15-04-11"),
        (3, "Feb-01-2023-15-11-00"),
        (3, "Feb-01-2023-15-18-14"),
        (3, "Feb-01-2023-15-24-59"),
        (3, "Feb-01-2023-15-33-39"),
        (0, "Feb-01-2023-15-41-48"),
        (0, "Feb-01-2023-15-48-21"),
        (2, "Feb-01-2023-15-53-54"),
        (2, "Feb-01-2023-16-01-49"),
        (0, "Feb-01-2023-16-08-28"),
        (2, "Feb-01-2023-16-14-50"),
        (3, "Feb-01-2023-16-25-08"),
        (1, "Feb-01-2023-17-24-30"),
        (0, "Feb-01-2023-17-31-27"),
        (1, "Feb-01-2023-17-39-31"),
        (2, "Feb-01-2023-17-45-45"),
        (0, "Feb-01-2023-17-53-59"),
    ]),
    ("/scratch0/zhang401local/plant_data/experiments/Feb-02-2023",[
        (2, "Feb-02-2023-09-01-36"),
        (2, "Feb-02-2023-09-08-37"),
        (0, "Feb-02-2023-09-15-15"),
        
    ]),
]
sparse_full_folders = [
    ("/scratch0/zhang401local/plant_data/experiments/Feb-02-2023",[
        (0, "Feb-02-2023-09-28-55"),
        (2, "Feb-02-2023-09-36-48"),
        (0, "Feb-02-2023-09-43-48"),
        (3, "Feb-02-2023-09-49-46"),
        (1, "Feb-02-2023-09-57-47"),
        (2, "Feb-02-2023-10-04-07"),
        (1, "Feb-02-2023-10-12-24"),
        (0, "Feb-02-2023-10-22-46"),
        (3, "Feb-02-2023-10-30-21"),
        (2, "Feb-02-2023-10-37-37"),
        (2, "Feb-02-2023-10-57-04"),
        (3, "Feb-02-2023-11-04-46"),
        (2, "Feb-02-2023-11-13-15"),
        (0, "Feb-02-2023-11-20-43"),
        (1, "Feb-02-2023-11-41-37"),
        (1, "Feb-02-2023-11-48-17"),
        (1, "Feb-02-2023-11-56-21"),
        (3, "Feb-02-2023-12-03-52"),
        (0, "Feb-02-2023-12-10-56"),
        (3, "Feb-02-2023-12-17-51"),
        (0, "Feb-02-2023-12-28-01"),
        (0, "Feb-02-2023-12-35-04"),
        (2, "Feb-02-2023-12-42-33"),
        (1, "Feb-02-2023-12-50-34"),
        (3, "Feb-02-2023-12-58-40"),
        (0, "Feb-02-2023-13-10-24"),
        (2, "Feb-02-2023-13-16-46"),
        (1, "Feb-02-2023-13-24-32"),
        (0, "Feb-02-2023-13-33-01"),
        (1, "Feb-02-2023-13-41-51"),
        (1, "Feb-02-2023-13-55-01"),
        (3, "Feb-02-2023-14-02-38"),
        (1, "Feb-02-2023-14-16-27"),
        (3, "Feb-02-2023-14-24-11"),
        (0, "Feb-02-2023-14-31-50"),
        (2, "Feb-02-2023-14-38-35"),
        (2, "Feb-02-2023-14-46-42"),
        (3, "Feb-02-2023-14-55-09"),
        (3, "Feb-02-2023-15-04-19"),
        (2, "Feb-02-2023-15-33-55"),
    ]),
]

def plot_all(folders, previous_results=None):
    model_trials = collections.defaultdict(list)
    for model_type, file_path_full in folders:
        area_changes = space_reveal_ratio(file_path_full)
        
        if previous_results:
            previous_trial_num = len(previous_results[model_type])
        else:
            previous_trial_num = 0
        
        trial_num = len(model_trials[model_type]) + previous_trial_num
        name = f"{model_type}-{trial_num}"
        model_trials[model_type].append((name, area_changes))
    
    return dict(model_trials)

def get_saved_information(root):
    policy_output_dict = {}
    
    for file in os.listdir(root):
        if file.endswith("policy_output.pkl"):
            file_parts = file.split("_")
            try:
                i_traj = int(file_parts[0])
                t = int(file_parts[1])
            except:
                continue

            policy_output = pickle.load(
                open(os.path.join(root, f'{i_traj:05}_{t:05}_policy_output.pkl'), "rb")
            )
            if os.path.exists(os.path.join(root, f'{i_traj:05}_{t:05}_policy_output_additional.pkl')):
                additional_info = pickle.load(open(os.path.join(root, f'{i_traj:05}_{t:05}_policy_output_additional.pkl'), "rb"))
                policy_output.update(additional_info)
            policy_output_dict[(i_traj, t)] = policy_output

    return policy_output_dict

def get_returned_infos(root):
    returned_infos_dict = {}
    for file in os.listdir(root):
        if file.endswith("_robot.pkl"):
            file_parts = file.split("_")
            try:
                i_traj = int(file_parts[0])
                t = int(file_parts[1])
            except:
                continue

            returned_infos = pickle.load(
                open(os.path.join(root, f'{i_traj:05}_{t:05}_robot.pkl'), "rb")
            )

            returned_infos_dict[(i_traj, t)] = returned_infos
    return returned_infos_dict

def space_reveal_ratio(root):
    policy_output_dict = {} #get_saved_information(root)
    i_traj = 0
    for t in range(10):
        try:
            policy_output = pickle.load(
                open(os.path.join(root, f'{i_traj:05}_{t:05}_policy_output.pkl'), "rb")
            )
            policy_output_dict[(i_traj, t)] = policy_output
        except:
            break
    config = YamlConfig(os.path.join(root, "policy_config.yaml"))
    config = dict(config.config)
    x_start_calculate, x_end_calculate, y_start_calculate, y_end_calculate = \
        np.asarray(config["calculate_bounds_in_space_revealed_map"]).astype(int)
    total_area = (x_end_calculate - x_start_calculate + 1) * (y_end_calculate - y_start_calculate+ 1)
    
    area_changes = [0]
    start_total_area = 0.0
    keys = sorted(list(policy_output_dict.keys()))
    new_areas = 0
    old_new_areas = policy_output_dict[(0,0)]['space_revealed_map_before_action'][y_start_calculate:y_end_calculate+1, \
                                                       x_start_calculate:x_end_calculate+1]
    #np.zeros((y_end_calculate - y_start_calculate+ 1, x_end_calculate - x_start_calculate + 1))
    new_areas_changes = [0]
    
    new_after_area = 0
    
    for k in keys:
        policy_output = policy_output_dict[k]
        space_revealed_map_before_action = policy_output["space_revealed_map_before_action"]
        if "new_space_revealed_map_gt_manual" in policy_output:
            new_space_revealed_map_gt = policy_output["new_space_revealed_map_gt_manual"]
        else:
            print(":: Warning using new_space_revealed_map_gt instead of manual! ", root, k)
            new_space_revealed_map_gt = policy_output["new_space_revealed_map_gt"]
        
        if k[1] == 0:
            start_total_area = np.sum(space_revealed_map_before_action[y_start_calculate:y_end_calculate+1, \
                                                       x_start_calculate:x_end_calculate+1])
            # print(root, start_total_area * (0.2*0.2))
            # print(start_total_area/total_area)
        
        newly_added = new_space_revealed_map_gt[y_start_calculate:y_end_calculate+1, \
                                                       x_start_calculate:x_end_calculate+1] - old_new_areas
        old_new_areas = np.maximum(
            old_new_areas,
            new_space_revealed_map_gt[y_start_calculate:y_end_calculate+1, \
                                                       x_start_calculate:x_end_calculate+1],
        )
        new_area = np.sum(newly_added[newly_added > 0])
        before_area = np.sum(space_revealed_map_before_action[y_start_calculate:y_end_calculate+1, \
                                                       x_start_calculate:x_end_calculate+1])
        after_area = np.maximum(space_revealed_map_before_action, new_space_revealed_map_gt)
        after_area = np.sum(after_area[y_start_calculate:y_end_calculate+1, \
                                                       x_start_calculate:x_end_calculate+1])
        
        new_after_area += (after_area - before_area)
        area_changes.append(new_after_area / total_area)
        
        # after_area -= start_total_area
        new_areas += new_area
        # area_changes.append(after_area/total_area)
        new_areas_changes.append(new_areas/total_area)
    return np.asarray(area_changes) * total_area, np.asarray(new_areas_changes) * total_area, start_total_area, total_area

def get_all_start_areas(full_folders, total_area = 82070):
    full_folders = [(x[0], os.path.join(droot, x[1])) for droot, L in full_folders for x in L]
    good = False
    start_total_areas = [] #collections.defaultdict(list)
    for model_type, folder in full_folders:        
        while not good:
            try:
                
                start_total_area = np.load(os.path.join(folder, "start_total_area.npy"))[0]
                start_total_area *= total_area * (0.2*0.2)
                start_total_areas.append(start_total_area) # [model_type]
                good = True
            except:
                area_changes, new_areas_changes, start_total_area = space_reveal_ratio(folder)
                np.save(os.path.join(folder, "area_changes_new_without_space_at_0.npy"), new_areas_changes)
                np.save(os.path.join(folder, "area_changes.npy"), area_changes)
                np.save(os.path.join(folder, "start_total_area.npy"), np.array([start_total_area]))
                
                # area_changes, new_areas_changes = space_reveal_ratio(folder)
                # np.save(os.path.join(folder, "area_changes_new.npy"), new_areas_changes)
                # np.save(os.path.join(folder, "area_changes.npy"), area_changes)
                print(folder, start_total_area * total_area * (0.2 * 0.2))
    return start_total_areas

def get_results(full_folders, T=11, use_cm_2 = False, total_area = 82070):
    full_folders = [(x[0], os.path.join(droot, x[1])) for droot, L in full_folders for x in L]
    results = collections.defaultdict(list)

    for model_type, folder in full_folders:
        good = False
        # area_changes, new_areas_changes, start_total_area = space_reveal_ratio(folder)
        # np.save(os.path.join(folder, "area_changes_new_without_space_at_0.npy"), new_areas_changes)
        # np.save(os.path.join(folder, "area_changes.npy"), area_changes)
        # np.save(os.path.join(folder, "start_total_area.npy"), np.array([start_total_area]))
        # print(folder, start_total_area * total_area * (0.2 * 0.2))
        # area_changes *= total_area * (0.2 * 0.2)
        # new_areas_changes *= total_area * (0.2 * 0.2)
        # start_total_area *= total_area * (0.2 * 0.2)
        # trial_num = len(results[model_type]) #+ previous_trial_num
        # name = f"{model_type}-{trial_num}"
        # results[model_type].append((name, new_areas_changes, start_total_area))
        while not good:
            try:
                result_path = os.path.join(folder, "area_changes_new_without_space_at_0.npy")
                new_areas_changes = np.load(result_path)
                result_path2 = os.path.join(folder, "area_changes.npy")
                area_changes = np.load(result_path2)
                start_total_area = np.load(os.path.join(folder, "start_total_area.npy"))[0]

                
                if use_cm_2:
                    area_changes *= total_area
                    area_changes *= (0.2 * 0.2)

                    new_areas_changes *= total_area * (0.2 * 0.2)
                    start_total_area *= total_area * (0.2 * 0.2)

                trial_num = len(results[model_type]) #+ previous_trial_num
                name = f"{model_type}-{trial_num}"
                results[model_type].append((name, new_areas_changes, start_total_area))
                good=True
            except:
                area_changes, new_areas_changes, start_total_area, total_area = space_reveal_ratio(folder)
                np.save(os.path.join(folder, "area_changes_new_without_space_at_0.npy"), new_areas_changes)
                np.save(os.path.join(folder, "area_changes.npy"), area_changes)
                np.save(os.path.join(folder, "start_total_area.npy"), np.array([start_total_area]))

                print(f"{folder}: total area = {total_area}")

                # print(folder, start_total_area * total_area * (0.2 * 0.2))
                # area_changes, new_areas_changes = space_reveal_ratio(folder)
                # np.save(os.path.join(folder, "area_changes_new.npy"), new_areas_changes)
                # np.save(os.path.join(folder, "area_changes.npy"), area_changes)
                # print("calculate: ", os.path.join(folder, "area_changes_new.npy"))


    results_array = {}
    for model_type, L in results.items():
        start_total_areas = [x[2] for x in L]
        lengths = [len(x[1]) for x in L]
        
        min_length = min(lengths)
        if T+1 <= min_length:
            L = [x[1][:T+1] for x in L]
        else:
            L = [np.pad(x[1], (0, T-len(x[1]))) for x in L]
    
        values = np.stack(L) #(N, 11)
        print(model_type, values.shape)
        # pop_mean, pop_std = repeated_mean(values, trials=trials)
        
        values_mean = values.mean(axis=0)
        values_std = values.std(axis=0)
        results_array[model_type] = (values, values_mean, values_std, np.mean(start_total_areas))
    return results_array

def prob_improvement(method1, method2):
    N = len(method1)
    K = len(method2)
    
    def funcs(x,y):
        
        if y < x:
            return 1
        if x == y:
            return 0.5
        return 0
    acc = 0
    for i in range(N):
        for j in range(K):
            acc += funcs(method1[i], method2[j])
    return acc / (N * K)

def plot_lines(data_dict, model_types_to_plot = [], idx_to_plot = collections.defaultdict(list), plot_mean=False, 
               plot_std=False, plot_all=True):
    fig, ax = plt.subplots(figsize=(12,8))
    markers = ["o", "v", "^", "D", "X", "<", "s", "p", "P", "*",">"]
    colors = ["k","b","g","r","c","m","y","gray"]
    keys = list(data_dict.keys())
    keys = sorted(keys)
    for idx, model_type in enumerate(keys):
        if len(model_types_to_plot) ==0 or model_type not in model_types_to_plot:
            continue
        lines = data_dict[model_type]
        values, means, stds = lines[:3]

        mean_to_plot = means
        std_to_plot = stds
            
        if plot_mean:
            # linestyle='--', marker=markers[model_type%len(markers)], 
            ax.plot(mean_to_plot, 
                    color=colors[model_type%len(colors)], label=model_type)
        if plot_std:
            
            ax.fill_between(np.arange(len(means)), mean_to_plot - std_to_plot, 
                            mean_to_plot + std_to_plot, color=colors[model_type%len(colors)], 
                            label=model_type, alpha=0.4)
        
        if plot_all:
            for j, one_line in enumerate(values):
                if len(idx_to_plot[model_type]) > 0 and j not in idx_to_plot[model_type]:
                    continue
                name = f"{model_type}_{j}"
                ax.plot(one_line, linestyle='--', \
                    marker=markers[j%len(markers)], alpha=0.4, color=colors[model_type%len(colors)], label=name)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig


default_method_names = {
    0 : "Tiling",
    1 : "VineReveal \n w/ hanging-spaghetti",
    2: "VineReveal \n w/ SRPNet No Image",
    3: "VineReveal \n w/ SRPNet",
}

def plot_lines_bootstrap(
        data_dict, 
        plot_mean=False, 
        plot_std=False, 
        plot_all=True, 
        re_sample=1000, 
        show_legend=False, 
        xlabel="Time Step", 
        ylabal="Space Revealed by Actions (#pixels)", 
        xlabel_fontsize=24,
        ylabel_fontsize=24,
        x_ticks_args = [0, 11, 1],
        y_ticks_args=[0,2500,500], 
        x_ticks_args_list = None,
        y_ticks_args_list = None,
        user_method_names=None,
        figsize=(8,6),
):
    fig, ax = plt.subplots(figsize=figsize)
    # legendFig = plt.figure("Legend plot")

    if user_method_names is None:
        method_names = default_method_names
    else:
        method_names = user_method_names
    

    markers = ["o", "v", "X", "+","*","^", "D",  "<", "s", "p",  ">"]
    colors = ["k","b","g","r","c","m","y","gray"]
    keys = list(data_dict.keys())
    keys = sorted(keys)
    bootstrap_info = {}
    lines_models = []
    names_models = []
    rng = np.random.default_rng()
    for idx, model_type in enumerate(keys):
        lines = data_dict[model_type]
        values = lines[0]
        stds = np.zeros((2, len(values[0])))
        mean_to_plot = np.zeros(len(values[0]))
        for step_idx in range(len(values[0])):
            if step_idx == 0:
                continue
            this_step_values = values[:,step_idx]
            # # print(model_type, this_step_values)
            # my_samples = []
            # for _ in range(re_sample):
            #     selected_mask = np.random.choice(len(this_step_values),len(this_step_values),replace=True)
                
            #     selected_values = this_step_values[selected_mask] # (subset_num, num_steps)
            #     my_samples.append(selected_values.mean())
            # low, high = sms.DescrStatsW(my_samples).tconfint_mean(alpha=0.01)

            low, high = st.t.interval(0.95, df=len(this_step_values)-1, loc=np.mean(this_step_values), scale=st.sem(this_step_values)) 
            # print(low,high)
            res = bootstrap((this_step_values,), np.mean, confidence_level=0.95,random_state=rng)
            
            low = res.confidence_interval.low
            high = res.confidence_interval.high
            # print(res.confidence_interval.low, res.confidence_interval.high)

            stds[0][step_idx] =  low
            stds[1][step_idx] =  high
            mean_to_plot[step_idx] = np.mean(this_step_values) #np.mean(this_step_values) #
        
            # print(tuple([np.array([v]).reshape(1,-1) for v in this_step_values]))
            # btstrp = bootstrap((np.stack(my_samples_not_meaned),), 
            #                    np.mean, axis=-1, confidence_level=0.95,
            #              random_state=1, method='percentile')
            # btstrp2 = bootstrap((this_step_values,), np.std, confidence_level=0.95,
            #              random_state=1, method='percentile')
            # print(low,high)
            # print(btstrp.confidence_interval.low)
            # print(btstrp.confidence_interval.high)
            # print("\n")
        
        bootstrap_info[model_type] = (mean_to_plot, stds)
        method_name = method_names[model_type]
        if plot_mean:
        
            label_name = f"{method_name}"
            # marker=markers[model_type%len(markers)],
            line = ax.plot(mean_to_plot,  linestyle='--', marker=markers[model_type%len(markers)],  markersize=10,
                    color=colors[model_type%len(colors)], label=label_name)
            lines_models.append(line)
            names_models.append(label_name)

        if plot_std:
            
            label_name = f"95%CI {method_name}"
            ax.fill_between(np.arange(len(mean_to_plot)), stds[0], 
                            stds[1], color=colors[model_type%len(colors)], 
                            alpha=0.2)
        
        if plot_all:
            for j, one_line in enumerate(values):
                name = f"{model_type}_{j}"
                ax.plot(one_line,  \
                    marker=markers[j%len(markers)], alpha=0.4, color=colors[model_type%len(colors)], 
                        label=name)
    if show_legend:
        # legendFig.legend(lines_models, names_models, loc='center')
        # legendFig.savefig('/home/zhang401local/Documents/legend.png')
        ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0, -0.5), handlelength=3, fontsize=21)
        # ax.legend(loc='lower center', ncol=4, bbox_to_anchor=(0, -0.5), handlelength=3, fontsize=21) # 
    ax.set_xlabel(xlabel, fontdict = {'fontsize': xlabel_fontsize})
    ax.set_ylabel(ylabal, fontdict = {'fontsize': ylabel_fontsize})
    # ax.set_title(title, fontdict = {'fontsize': 18})
    if x_ticks_args_list:
        ax.xaxis.set_ticks(x_ticks_args_list)
    else:
        ax.xaxis.set_ticks(list(np.arange(x_ticks_args[0], x_ticks_args[1], x_ticks_args[2])) )
    if y_ticks_args_list:
        ax.set_yticks(list(np.arange(y_ticks_args[0], y_ticks_args[1], y_ticks_args[2])),y_ticks_args_list)
    else:
        ax.yaxis.set_ticks(list(np.arange(y_ticks_args[0], y_ticks_args[1], y_ticks_args[2])) ) #+ [2000,2500]
    
    ax.tick_params(axis='both', which='major', labelsize=21)

    return bootstrap_info, fig, None

def plot_poI_bootstrap(method1_model_type, method1, data_dict, re_sample=1000):
    fig, ax = plt.subplots(figsize=(12,8))
    markers = ["o", "v", "^", "D", "X", "<", "s", "p", "P", "*",">"]
    colors = ["k","b","g","r","c","m","y","gray"]
    bar_data = {}
    keys = list(data_dict.keys())
    keys = sorted(keys)
    for idx, model_type in enumerate(keys):
        if model_type == method1_model_type:
            continue
        lines = data_dict[model_type]
        values = lines[0]

        model_res = np.zeros((len(values[0])-1, 3))
        
        for step_idx in range(len(values[0])):
            if step_idx == 0:
                continue
            this_step_values = values[:,step_idx]
            
            poI = prob_improvement(method1[:,step_idx], this_step_values)
            
            my_samples = []
            
            for _ in range(re_sample):
                selected_mask = np.random.choice(len(this_step_values),len(this_step_values),replace=True) 
                selected_values = this_step_values[selected_mask]
                
                poI_i = prob_improvement(method1[:,step_idx], selected_values)
                my_samples.append(poI_i)
                
            low, high = sms.DescrStatsW(my_samples).tconfint_mean()
            model_res[step_idx-1] = np.array([poI, low, high])
        
        bar_data[model_type] = model_res
    other_methods = list(bar_data.keys())
    other_methods = sorted(other_methods)
    y_min = 2
    y_max = -1
    for model_type in other_methods:
        name = model_type
        this_model_data = bar_data[model_type]
        y_min = min(this_model_data.min(), y_min)
        y_max = max(this_model_data.max(), y_max)
        num_steps = len(this_model_data)
        poI = this_model_data[:,0]
        ax.plot(np.arange(1, num_steps+1), poI, linestyle='--', \
                    marker=markers[model_type%len(markers)], alpha=0.4, 
                color=colors[model_type%len(colors)], 
                        label=name)
        ax.errorbar(np.arange(1, num_steps+1), poI, xerr=0, 
                 yerr=np.abs(this_model_data[:,1:].T - poI), 
                 fmt=markers[model_type%len(markers)], 
                    color=colors[model_type%len(colors)], 
                    ecolor=colors[model_type%len(colors)], 
                    capsize=3, label=name)
    ax.set_ylim([y_min-0.05, y_max+0.05])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return bar_data, fig