import numpy as np
import cv2
import glob
from skimage.feature import hog

def get_features(img, method):

    # Convert images to 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 128), interpolation = cv2.INTER_AREA)
    gray_arr = np.asarray(gray)

    # Ratio hair feature
    img_white = np.ones((128, 96), dtype = np.uint8)
    count = 0
    for i in range(0,128):
        for j in range(0,96):
            if gray_arr[i][j] <= 51 :
                img_white[i][j] = 0
                count += 1
    ratio_h = [count / (128*96)]

    # HOG features
    hog_features, hog_image = hog(img_white, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize = True)
    orient_features = np.zeros((4))
    for i in range(len(hog_features)):
        orient_features[i%4] += hog_features[i]

    # Concatenate histograms features and HOG features
    merged_features = np.concatenate((orient_features, ratio_h))

    if method == "Ratio hair":
        return ratio_h
    elif method == "HOG":
        return orient_features
    else:
        return merged_features
    
def prediction(img, method):

    #Intialize params
    ratio_h_path = "../data/ratio_h_features/"
    hog_path = "../data/HOG_features/"
    all_features_path = "../data/all_features/"

    val = []
    idx = []

    current_features = get_features(img, method)

    if method == "Ratio hair":
        for path in glob.glob(f'{ratio_h_path}/*'):
            ratio_h_tmp = np.load(path)
            dist = np.linalg.norm(ratio_h_tmp - current_features)
            val.append(dist)
            idx.append(path)

    elif method == "HOG":
        for path in glob.glob(f'{hog_path}/*'):
            hog_tmp = np.load(path)
            dist = np.linalg.norm(hog_tmp - current_features)
            val.append(dist)
            idx.append(path)

    else:
        for path in glob.glob(f'{all_features_path}/*'):
            all_features_tmp = np.load(path)
            dist = np.linalg.norm(all_features_tmp - current_features)
            val.append(dist)
            idx.append(path)

    nearest_img = np.argsort(val)[:3]
    res_nu = 0
    for img_path in nearest_img:
        if 'nu' in idx[img_path]:
            res_nu += 1
    
    return "Nu" if res_nu > 1 else "Nam"