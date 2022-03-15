#from PIL import Image
import imageio
import pathlib
import glob
import os
import time
import click

@click.command()
@click.option(
    "--fp_in",
    type=pathlib.Path,
    required=True,
    help="Path to the folder containing the images",
)
@click.option(
    "--fp_out",
    type=pathlib.Path,
    required=True,
    help="Path to the output_files",
)
@click.option(
    "--fps",
    type=int,
    required=False,
    default=60,
    help="Path to the folder containing the images",
)
@click.option(
    "--save_each_files",
    type=int,
    required=False,
    default=144,
    help="Increment to save a gif each N files, if 0 only the last will be saved",
)
def main(fp_in, fp_out, fps, save_each_files):

    fp_in = fp_in / '*'

    # List files
    list_of_files = filter( os.path.isfile,
                            glob.glob(str(fp_in)) )
    # Sort list of files based on last modification time in ascending order
    filenames = sorted( list_of_files,
                            key = os.path.getmtime)

    # Parse files
    images = []
    for i, filename in enumerate(filenames):
        images.append(imageio.imread(filename))
        if save_each_files != 0 and i % save_each_files == 0:
            file_out = str(fp_out) + "_" + str(i) + '.gif'
            imageio.mimsave(file_out, images, fps=fps)
            
    # N seconds
    print(len(filenames), len(filenames) // fps)
    file_out = str(fp_out) + "_all" + '.gif'
    imageio.mimsave(file_out, images, fps=fps)

if __name__ == "__main__":
    main()