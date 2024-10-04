import cv2
import numpy as np
import maxflow
from sklearn.mixture import GaussianMixture

GAMMA = 50
K = 5

def getPixelArray(image):
    """ Flatten a 3D image (H x W x C) into a 2D array (num_pixels x 3). """
    return image.reshape(-1, 3)

def fitGMM(pixels, mask):
    """ Fit a GMM to the pixels in the mask. """
    pixels_in_class = pixels[mask]

    gmm = GaussianMixture(n_components=K, covariance_type='full')
    gmm.fit(pixels_in_class)
    
    return gmm

def calculateWeights(gmm, pixels):
    """ Calculate the pixel-wise weights using the GMM model. """
    log_probs = gmm.score_samples(pixels)  # Get log-likelihood of each pixel
    return -log_probs

def calculateNeighborhoodWeights(image):
    """ Calculate the weights between neighboring pixels based on color difference. """
    h, w = image.shape[:2]

    
    down = np.sum((image[:-1, :, :] - image[1:, :, :]) ** 2, axis=2)
    right = np.sum((image[:, :-1, :] - image[:, 1:, :]) ** 2, axis=2)

    down_weights = GAMMA * np.exp(-0.5 * down / np.mean(down))
    right_weights = GAMMA * np.exp(-0.5 * right / np.mean(right))

    return down_weights, right_weights

def addGraphEdges(graph, nodeids, down_weights, right_weights):
    """ Add edges to the graph for neighboring pixels. """
    h, w = nodeids.shape

    
    down_weights_padded = np.pad(down_weights, ((0, 1), (0, 0)), mode='constant', constant_values=0)
    right_weights_padded = np.pad(right_weights, ((0, 0), (0, 1)), mode='constant', constant_values=0)

   
    structure = np.array([[0, 1, 0], 
                          [1, 0, 1], 
                          [0, 1, 0]])  

   
    graph.add_grid_edges(nodeids, down_weights_padded, structure=structure, symmetric=True)
    graph.add_grid_edges(nodeids, right_weights_padded, structure=structure, symmetric=True)

def grabCut(image, rect, num_iters=5):
    """ Perform GrabCut segmentation using GMM and max-flow. """
    h, w = image.shape[:2]

    
    x, y, w_box, h_box = rect
    mask = np.zeros((h, w), dtype=np.uint8)  # Initial mask, 0 for background
    mask[y:y+h_box, x:x+w_box] = 1  # 1 for foreground (initial guess)

    pixels = getPixelArray(image)

    for _ in range(num_iters):
        
        # Fit GMMs to foreground and background pixels
        fg_gmm = fitGMM(pixels, mask.flatten() == 1)
        bg_gmm = fitGMM(pixels, mask.flatten() == 0)

        # Calculate pixel-wise weights for foreground and background
        fg_weights = calculateWeights(fg_gmm, pixels)
        bg_weights = calculateWeights(bg_gmm, pixels)

      
        fg_weights = fg_weights.reshape(h, w)
        bg_weights = bg_weights.reshape(h, w)

       
        down_weights, right_weights = calculateNeighborhoodWeights(image)

       
        graph = maxflow.Graph[float]()
        nodeids = graph.add_grid_nodes((h, w))

     
        addGraphEdges(graph, nodeids, down_weights, right_weights)

     
        graph.add_grid_tedges(nodeids, fg_weights, bg_weights)

      
        graph.maxflow()

      
        new_mask = graph.get_grid_segments(nodeids).astype(np.uint8)

        mask = new_mask  

    # Return both foreground and background masks
    fgMask = mask
    bgMask = np.logical_not(mask)
    
    return fgMask, bgMask

def applyGrabCut(image, rect):
    """ Apply GrabCut and return foreground mask. """
    fgMask, bgMask = grabCut(image, rect)
    
    # Create segmented image based on the mask
    result = image.copy()
    result[bgMask] = 0  
    
    return fgMask, bgMask, result
