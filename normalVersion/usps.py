import os
from PIL import Image
import numpy as np

def usps(dir_number = 10,max_files_per_dir = 1000):  # Set your limit here
    print("Loading USPS data...")

    path_to_data = "./USPSdata/Numerals/"

    img_list = os.listdir(path_to_data)

    sz = (28,28)
    validation_usps = []
    validation_usps_label = []

    for i in range(dir_number):
        label_data = path_to_data + str(i) + '/'
        img_list = os.listdir(label_data)
        file_count = 0
        for name in img_list:
            if '.png' in name:
                img = Image.open(label_data + name)
                img = img.resize(sz)
                img_array = np.array(img).flatten()  # Convert the image to a 1D array
                validation_usps.append(img_array)
                validation_usps_label.append(i)  # Add the label (i.e., the digit)

                file_count += 1
                if file_count >= max_files_per_dir:
                    break


    # Convert the list of arrays to a 2D array
    validation_usps = np.array(validation_usps)
    print(validation_usps.shape)
    print("Done loading USPS data.")
    return validation_usps, validation_usps_label

    # Now validation_usps is a 2D array where each row is an image
