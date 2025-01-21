import rasterio as rio
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pywecs

def main():
    print('\n\nRunning PyWECS...\n')
    
    # read data paths
    directory = 'data'
    files = [directory + "/" + f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.startswith
    ('raster')]

    # regex function for ordering files
    def extract_number(s):
        match = re.search(r'\d+', s)
        return int(match.group()) if match else float('inf')

    sorted_files = sorted(files, key=extract_number)

    # initialize X
    X = []

    # read data and concatenate raster cube
    start = time.time()
    for n, i in enumerate(sorted_files):
        temp = rio.open(i)
        X.append(temp.read())
        temp.close()
        print(str(n+1)+"/"+str(len(sorted_files))+": "+sorted_files[n], end="\r")
    X = np.concatenate(X)
    end = time.time()
    elapsed_time = end - start
    print(f"Raster concatenation elapsed time: {elapsed_time:.2f} seconds.")

    # reduce image for wavelet
    X = X[:,:,:-3]

    # run WECS
    start = time.time()
    wecs = pywecs.WECS(X)
    end = time.time()
    elapsed_time = end - start
    print(f"WECS elapsed time: {elapsed_time:.2f} seconds.")

    # Otsu threshold segmentation
    bin_wecs = pywecs.segment_otsu(wecs)

    print('\n\nPlotting...\n\n')

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    axes[0].imshow(wecs, cmap='cividis', interpolation='nearest')
    axes[0].set_title('WECS')
    axes[0].axis('off')  
    axes[1].imshow(bin_wecs, cmap='cividis')
    axes[1].set_title('Otsu-WECS')
    axes[1].axis('off')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()
    

if __name__ == '__main__':
    main()