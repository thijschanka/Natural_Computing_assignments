#from PIL import Image
import imageio
import glob
import os
import time

dir_name = 'assignment3_outputs//high_bc_bg//'
fp_out = "assignment3_outputs//high_bc_bg_{}.gif"
fps = 60
save_each_files = 144

# List files
list_of_files = filter( os.path.isfile,
                        glob.glob(dir_name + '*') )
# Sort list of files based on last modification time in ascending order
filenames = sorted( list_of_files,
                        key = os.path.getmtime)

# Parse files
images = []
for i, filename in enumerate(filenames):
    images.append(imageio.imread(filename))
    if i % save_each_files == 0:
        imageio.mimsave(fp_out.format(str(i)), images, fps=fps)
        
# N seconds
print(len(filenames), len(filenames) // fps)

imageio.mimsave(fp_out.format("all"), images, fps=fps)