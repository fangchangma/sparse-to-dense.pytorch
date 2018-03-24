import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

cmap = plt.cm.jet

def merge_into_row(input, target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth = np.squeeze(target.cpu().numpy())
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:,:,:3] # H, W, C
    pred = np.squeeze(depth_pred.data.cpu().numpy())
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    pred = 255 * cmap(pred)[:,:,:3] # H, W, C
    img_merge = np.hstack([rgb, depth, pred])
    
    # img_merge.save(output_directory + '/comparison_' + str(epoch) + '.png')
    return img_merge

def add_row(img_merge, row):
    return np.vstack([img_merge, row])

def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)