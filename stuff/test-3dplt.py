import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import os

# --- configuration for our animation ---
num_frames = 500 # how many images we want!
output_dir = './3d-test/' # where all the cute images will go!

# --- static cube properties (these won't change per frame) ---
num_cubes_side = 8 # an 8x8 grid for our city
cube_width = 0.8    # base width of each cube
padding = 0.2       # the space between each cube

# 1. create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 2. create the positions for each cube's base (calculated once)
x_coords = np.arange(num_cubes_side) * (cube_width + padding)
y_coords = np.arange(num_cubes_side) * (cube_width + padding)

X_pos, Y_pos = np.meshgrid(x_coords, y_coords)

x_flat = X_pos.flatten()
y_flat = Y_pos.flatten()
z_flat = np.zeros_like(x_flat) # all cubes start on the ground (z=0)

dx_flat = np.full_like(x_flat, cube_width)
dy_flat = np.full_like(y_flat, cube_width)

# 3. prepare for the gradient color calculation (normalize positions once)
min_x, max_x = x_flat.min(), x_flat.max()
min_y, max_y = y_flat.min(), y_flat.max()

x_norm = (x_flat - min_x) / (max_x - min_x)
y_norm = (y_flat - min_y) / (max_y - min_y)

gradient_values = (x_norm + y_norm) / 2.0 # for a diagonal gradient

# 4. define our base colors in RGB and convert them to HSV
initial_start_rgb = np.array([0.8, 0.6, 0.9]) # light purple
initial_end_rgb = np.array([1.0, 0.7, 0.8])  # soft pink

initial_start_hsv = mcolors.rgb_to_hsv(initial_start_rgb)
initial_end_hsv = mcolors.rgb_to_hsv(initial_end_rgb)

# now, loop through each frame to generate and save the images!
print(f"generating {num_frames} frames... this might take a bit! <3")
for i in range(num_frames):
    frame_num = i + 1 # for 1-based indexing in filenames
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- THIS IS THE CRUCIAL CHANGE! ---
    # 3. generate NEW random heights for each cube in EVERY frame!
    dz_flat = np.random.rand(len(x_flat)) * 4.5 + 0.5 # random heights between 0.5 and 5.0

    # calculate the hue shift for this frame
    current_hue_shift = (i * (1.0 / num_frames)) % 1.0

    # apply the hue shift to our base HSV colors
    shifted_start_hsv = initial_start_hsv.copy()
    shifted_end_hsv = initial_end_hsv.copy()

    shifted_start_hsv[0] = (shifted_start_hsv[0] + current_hue_shift) % 1.0
    shifted_end_hsv[0] = (shifted_end_hsv[0] + current_hue_shift) % 1.0

    # convert shifted HSV back to RGB
    shifted_start_rgb = mcolors.hsv_to_rgb(shifted_start_hsv)
    shifted_end_rgb = mcolors.hsv_to_rgb(shifted_end_hsv)

    # interpolate between the two shifted colors for each cube!
    frame_colors = []
    for val in gradient_values:
        interp_color = shifted_start_rgb * (1 - val) + shifted_end_rgb * val
        frame_colors.append(tuple(interp_color))

    # plot the 3D bars for this frame!
    ax.bar3d(x_flat, y_flat, z_flat, dx_flat, dy_flat, dz_flat, color=frame_colors, shade=True)

    # label your axes and give it a title!
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Random Height (Z)')
    ax.set_title(f'frame {frame_num:03d}')

    # adjust the view angle
    ax.view_init(elev=30, azim=-60)

    # save this frame!
    filename = os.path.join(output_dir, f'{frame_num:03d}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')

    # close the figure to free up memory!
    plt.close(fig)

print(f"\nall {num_frames} images saved in the '{output_dir}' folder! they should all be super random now! yay! -w-")
print("now you can use something like ffmpeg to stitch them into a video if you want!")
