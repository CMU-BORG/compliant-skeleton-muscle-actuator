import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from collections import deque
import matplotlib.pyplot as plt
import os
import json
import tkinter as tk
from tkinter import filedialog

def split_bgr_image(img, visualize=False):
    blue = np.zeros_like(img)
    green = np.zeros_like(img)
    red = np.zeros_like(img)
    blue[:,:,0] = img[:,:,0]
    green[:,:,1] = img[:,:,1]
    red[:,:,2] = img[:,:,2]
    if visualize:
        cv2.imshow("Blue", blue)
        cv2.imshow("Green", green)
        cv2.imshow("Red", red)
        cv2.waitKey(0)
    return blue, green, red

def morph_open(img, kernel_size=5, iterations=1):
    if kernel_size < 1:
        kernel_size = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opened

def get_contour_centroids(contours):
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            points = contour.reshape(-1, 2)
            cx = np.mean(points[:, 0])
            cy = np.mean(points[:, 1])
        centroids.append(np.int32((cx, cy)))
    return centroids

def get_cell_depths(surface_contour, cell_centroids):
    surface_contour = surface_contour.reshape(-1, 2)
    cell_depths = np.zeros(len(cell_centroids))
    nearest_points = np.zeros_like(cell_centroids)
    for i, centroid in enumerate(cell_centroids):
        distances = np.linalg.norm(surface_contour - centroid, axis=1) * px_to_um
        nearest_points[i] = surface_contour[np.argmin(distances)]
        cell_depths[i] = np.min(distances)
    return cell_depths, nearest_points

def update_main_contour(*args):    
    if update_params_from_trackbar:
        main_params['block_size'] = cv2.getTrackbarPos('Block Size', 'Main Contour Adjustment') 
        main_params['C'] = cv2.getTrackbarPos('C', 'Main Contour Adjustment') 
        main_params['kernel_size'] = cv2.getTrackbarPos('Kernel Size', 'Main Contour Adjustment') 
        main_params['iterations'] = cv2.getTrackbarPos('Iterations', 'Main Contour Adjustment') 

    
    img_bw = cv2.adaptiveThreshold(
        img_gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV,
        main_params['block_size']*2+3,
        main_params['C']+1)
    
    img_opened = morph_open(img_bw, kernel_size=main_params['kernel_size'], iterations=main_params['iterations'])
    contours, _ = cv2.findContours(img_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        result = img.copy()
        cv2.drawContours(result, [main_contour], -1, (0, 255, 0), 2)
        cv2.imshow("Main Contour Adjustment", result)
    return main_contour

def manhattan_distance_floodfill(binary_map, mask):
    """
    Compute Manhattan distance transform using iterative flood fill (4-way connectivity)
    :param binary_map: Binary image where contour pixels are 1 and others are 0
    :return: Distance map where each pixel contains Manhattan distance to nearest contour
    """
    h, w = binary_map.shape
    dist_map = np.zeros((h, w), dtype=np.uint32)  # Will store distances
    
    # Initialize queue with all contour points (distance = 0)
    q = deque()
    for y in range(h):
        for x in range(w):
            if binary_map[y, x] != 0 and mask[y,x] != 0:
                q.append((x, y, 0))  # (x, y, distance)
                dist_map[y, x] = 0
    
    # 4-way connectivity directions: right, left, down, up
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    # Process queue until all pixels are filled
    while q:
        x, y, d = q.popleft()
        
        # Check all neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < w and \
                0 <= ny < h and \
                binary_map[ny, nx] == 0 and \
                dist_map[ny, nx] == 0 and \
                mask[ny, nx] != 0:
                    new_dist = d + 1
                    print(new_dist)
                    dist_map[ny, nx] = new_dist
                    q.append((nx, ny, new_dist))
    
    return dist_map

def smooth_contour_via_mask(contour, img_shape, kernel_size=11, iterations = 2):
    """Smooth by converting to mask and applying morphological operations"""
    # Create binary mask
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, -1)  # Fill interior
    
    # Apply morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=iterations)
    
    # Find new contour
    new_contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(new_contours, key=cv2.contourArea)

def smooth_contour_coordinates(contour, sigma=3.0):
    """Smooth contour by applying Gaussian blur to coordinates"""
    # Convert contour points to float32
    points = contour.reshape(-1, 2).astype(np.float32)
    
    # Create 2D arrays for x and y coordinates
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1].reshape(-1, 1)
    
    # Apply Gaussian blur to X and Y coordinates
    x_smoothed = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma)
    y_smoothed = cv2.GaussianBlur(y, (0, 0), sigmaX=sigma)
    
    # Combine back into points array
    points[:, 0] = x_smoothed.ravel()
    points[:, 1] = y_smoothed.ravel()
    
    # Convert back to contour format
    return points.reshape(-1, 1, 2).astype(np.int32)

def process_image(fname, contour_params, global_settings, output_folder):
    """Process a single image and save all outputs"""
    # Load and resize image
    img = cv2.imread(fname)
    scale = global_settings['scale']
    smoothing = global_settings['smoothing']
    bin_width = global_settings['bin_width']
    px_to_um_base = global_settings['px_to_um_base']
    
    if scale != 1.0:
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)
    px_to_um = px_to_um_base / scale
    
    # Split channels
    nuclei_img, live_img, dead_img = split_bgr_image(img, visualize=False)
    nuclei_gray = cv2.cvtColor(nuclei_img, cv2.COLOR_BGR2GRAY)
    live_gray = cv2.cvtColor(live_img, cv2.COLOR_BGR2GRAY)
    dead_gray = cv2.cvtColor(dead_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute contour using stored parameters
    block_size = contour_params['block_size']
    C = contour_params['C']
    kernel_size = contour_params['kernel_size']
    iterations = contour_params['iterations']
    
    img_bw = cv2.adaptiveThreshold(
        img_gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV,
        block_size*2+3,
        C+1)
    
    img_opened = morph_open(img_bw, kernel_size=kernel_size, iterations=iterations)
    contours, _ = cv2.findContours(img_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"No contours found for {fname}. Skipping.")
        return
        
    raw_main_contour = max(contours, key=cv2.contourArea)
    
    # Smooth contour
    if smoothing > 0:
        main_contour = smooth_contour_coordinates(
            raw_main_contour, 
            sigma=smoothing*5*int(10*scale))
    else:
        main_contour = raw_main_contour

    # Create base name for output files
    base_name = os.path.splitext(os.path.basename(fname))[0]
    
    # Create depth map and interior mask
    depth_map = np.zeros_like(img_gray)
    cv2.drawContours(depth_map, [main_contour], -1, 1, 1)
    
    interior_mask = np.zeros_like(img_gray)
    cv2.drawContours(interior_mask, [main_contour], -1, 1, -1)
    
    # Save contour image
    contour_img = img.copy()
    cv2.drawContours(contour_img, [main_contour], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_contour.png"), contour_img)
    
    # Save interior mask
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_mask.png"), interior_mask * 255)
    
    # Compute Manhattan distance
    dist_map = manhattan_distance_floodfill(depth_map, interior_mask)
    if smoothing > 0:
        dist_map = cv2.GaussianBlur(
            dist_map.astype(np.float32), 
            (0,0), 
            sigmaX=smoothing*3*int(10*scale)
        ).astype(np.uint32)
    
    # Convert to physical units
    dist_map_um = dist_map * px_to_um
    
    # Extract interior pixels for analysis
    analysis_mask = (interior_mask != 0)  # Include both contour and interior
    analysis_distances = dist_map_um[analysis_mask]

    # Get intensities for each channel
    dead_intensities = dead_gray[analysis_mask]
    live_intensities = live_gray[analysis_mask]
    nuclei_intensities = nuclei_gray[analysis_mask]

    # Create DataFrame for analysis
    data = pd.DataFrame({
        'Distance': analysis_distances,
        'Dead': dead_intensities,
        'Live': live_intensities,
        'Nuclei': nuclei_intensities
    })
    
    # 1. Create and save unbinned line plot
    plt.figure(figsize=(12, 6))
    dead_means = data.groupby('Distance')['Dead'].mean()
    live_means = data.groupby('Distance')['Live'].mean()
    nuclei_means = data.groupby('Distance')['Nuclei'].mean()
    
    plt.plot(dead_means.index, dead_means.values, color='red', label='Dead')
    plt.plot(live_means.index, live_means.values, color='green', label='Live')
    plt.plot(nuclei_means.index, nuclei_means.values, color='blue', label='Nuclei')
    
    plt.xlabel("Distance from surface (um)")
    plt.ylabel("Mean Intensity")
    plt.title(f"{base_name} - Intensity vs Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{base_name}_line_plot.png"))
    plt.close()
    
    # 2. Create and save binned analysis
    max_dist = data['Distance'].max()
    bins = np.arange(0, max_dist + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Bin the data
    data['DistanceBin'] = pd.cut(data['Distance'], bins, labels=bin_centers, include_lowest=True)
    binned_data = data.groupby('DistanceBin').agg({
        'Dead': 'mean',
        'Live': 'mean',
        'Nuclei': 'mean'
    }).reset_index()
    
    # Normalize by first bin
    for channel in ['Dead', 'Live', 'Nuclei']:
        first_bin_value = binned_data[channel].iloc[0]
        if first_bin_value > 0:
            binned_data[f'{channel}_norm'] = binned_data[channel] / first_bin_value
        else:
            binned_data[f'{channel}_norm'] = binned_data[channel] / (first_bin_value + 1e-6)
    
    # Add bin information
    binned_data['Bin_Start'] = bins[:-1]
    binned_data['Bin_End'] = bins[1:]
    binned_data['Bin_Center'] = bin_centers
    
    # Create bin range strings for labeling
    binned_data['Bin_Range'] = [
        f"{int(start)}-{int(end)}" 
        for start, end in zip(binned_data['Bin_Start'], binned_data['Bin_End'])
    ]
    
    # Create fold change plot with bin range labels
    melted_norm = binned_data.melt(
        id_vars='Bin_Range', 
        value_vars=['Dead_norm', 'Live_norm', 'Nuclei_norm'],
        var_name='Channel', 
        value_name='Fold Change'
    )
    melted_norm['Channel'] = melted_norm['Channel'].str.replace('_norm', '')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Bin_Range', y='Fold Change', hue='Channel', data=melted_norm, 
                palette={'Dead': 'red', 'Live': 'green', 'Nuclei': 'blue'})
    plt.axhline(1, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Distance from surface (um)")
    plt.ylabel(f"Fold Change (Relative to {binned_data['Bin_Range'].iloc[0]}μm)")
    plt.title(f"{base_name} - Fold Change ({bin_width}um Bins)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{base_name}_fold_change.png"))
    plt.close()
    
    
    # Add bin information to DataFrame
    binned_data['Bin_Start'] = bins[:-1]
    binned_data['Bin_End'] = bins[1:]
    binned_data['Bin_Center'] = bin_centers
    
    # Reorder columns
    column_order = [
        'Bin_Start', 'Bin_End', 'Bin_Center',
        'Dead', 'Live', 'Nuclei',
        'Dead_norm', 'Live_norm', 'Nuclei_norm'
    ]
    binned_data = binned_data[column_order]
    
    # Save data files
    binned_data.to_csv(os.path.join(output_folder, f"{base_name}_binned.csv"), index=False)
    data.to_csv(os.path.join(output_folder, f"{base_name}_full_data.csv"), index=False)
    
    # Save parameters
    analysis_metadata = {
        'image_filename': fname,
        'scale_factor': scale,
        'px_to_um_base': px_to_um_base,
        'smoothing': smoothing,
        'bin_width_um': bin_width,
        'contour_params': contour_params
    }
    with open(os.path.join(output_folder, f"{base_name}_params.json"), 'w') as f:
        json.dump(analysis_metadata, f, indent=4)
    
    # Save distance map visualization
    if dist_map.dtype != np.float32:
        dist_map = dist_map.astype(np.float32)
    
    viz_map = np.zeros_like(dist_map, dtype=np.uint8)
    cv2.normalize(dist_map, viz_map, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored_viz = cv2.applyColorMap(viz_map, cv2.COLORMAP_CIVIDIS) 
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_distance_map.png"), colored_viz)
    
    print(f"Processed and saved outputs for: {base_name}")

def adjust_contour(img_gray, contour_params):
    """Interactive contour adjustment for a single image"""
    cv2.namedWindow('Main Contour Adjustment')
    cv2.createTrackbar('Block Size', 'Main Contour Adjustment', contour_params['block_size'], int(global_settings['scale']*500), lambda x: None)
    cv2.createTrackbar('C', 'Main Contour Adjustment', contour_params['C'], 20, lambda x: None)
    cv2.createTrackbar('Kernel Size', 'Main Contour Adjustment', contour_params['kernel_size'], 15, lambda x: None)
    cv2.createTrackbar('Iterations', 'Main Contour Adjustment', contour_params['iterations'], 10, lambda x: None)

    current_contour = None

    print("Adjust main contour parameters. Press 'n' when done.")
    while True:
        # Read trackbar positions
        block_size = cv2.getTrackbarPos('Block Size', 'Main Contour Adjustment')
        C = cv2.getTrackbarPos('C', 'Main Contour Adjustment')
        kernel_size = cv2.getTrackbarPos('Kernel Size', 'Main Contour Adjustment')
        iterations = cv2.getTrackbarPos('Iterations', 'Main Contour Adjustment')

        # Update contour parameters
        contour_params['block_size'] = block_size
        contour_params['C'] = C
        contour_params['kernel_size'] = kernel_size
        contour_params['iterations'] = iterations

        # Compute the contour
        img_bw = cv2.adaptiveThreshold(
            img_gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV,
            block_size*2+3,
            C+1)

        img_opened = morph_open(img_bw, kernel_size=kernel_size, iterations=iterations)
        contours, _ = cv2.findContours(img_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            current_contour = max(contours, key=cv2.contourArea)
            result = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(result, [current_contour], -1, (0, 255, 0), 2)
            cv2.imshow("Main Contour Adjustment", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            break

    cv2.destroyWindow('Main Contour Adjustment')
    return current_contour

def select_files():
    """Prompt user to select files or folders"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    choice = input("Select input type:\n1. Single PNG file\n2. Folder of PNG files\nEnter choice (1 or 2): ")
    
    if choice == '1':
        fname = filedialog.askopenfilename(
            title="Select PNG file",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        return [fname] if fname else []
    elif choice == '2':
        folder = filedialog.askdirectory(title="Select folder containing PNG files")
        if folder:
            return [
                os.path.join(folder, f) 
                for f in os.listdir(folder) 
                if f.lower().endswith('.png')
            ]
    return []

# Main script starts here
if __name__ == "__main__":
    # Global settings (adjust as needed)
    global_settings = {
        'scale': 0.25,
        'smoothing': 1,
        'bin_width': 100,  # in um
        'px_to_um_base': 0.57544
    }
    
    # Select input files
    file_list = select_files()
    if not file_list:
        print("No files selected. Exiting.")
        exit()
    
    # Phase 1: Collect parameters for each image
    contour_params_dict = {}
    for fname in file_list:
        print(f"\nAdjusting contour for: {os.path.basename(fname)}")
        
        # Load and resize image
        img = cv2.imread(fname)
        if global_settings['scale'] != 1.0:
            img = cv2.resize(img, (0,0), fx=global_settings['scale'], fy=global_settings['scale'])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Initialize parameters for this image
        contour_params = {
            'block_size': 1, 
            'C': 0, 
            'kernel_size': 3, 
            'iterations': 0
        }
        
        # Adjust contour
        adjust_contour(img_gray, contour_params)
        contour_params_dict[fname] = contour_params
    
    # Create output folder
    scale = global_settings['scale']
    smoothing = global_settings['smoothing']
    bin_width = global_settings['bin_width']
    output_folder = f"output_scale{scale}_smoothing{smoothing}_bin{bin_width}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nCreated output folder: {output_folder}")
    
    # Phase 2: Process all images
    for fname in file_list:
        print(f"\nProcessing: {os.path.basename(fname)}")
        process_image(
            fname, 
            contour_params_dict[fname], 
            global_settings, 
            output_folder
        )
    
    print("\nProcessing complete. All outputs saved in:", output_folder)