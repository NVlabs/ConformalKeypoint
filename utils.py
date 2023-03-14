import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
import torch

K     = 100
alpha = 0.8

def one_each(pred, thresh=0.0):
    # Postprocess frcnn: get at most one instance per class
    # Return: boxes and labels
    conf = pred['scores'] > thresh

    conf_scores = pred['scores'][conf]
    conf_boxes = pred['boxes'][conf].int()
    conf_labels = pred['labels'][conf].int()

    valid = torch.zeros_like(conf_labels).bool()
    unique_labels = torch.unique(conf_labels)
    for uni in unique_labels:
        p = (conf_labels==uni).nonzero(as_tuple=False).reshape(-1)
        valid[p[0]] = True

    pd_scores = conf_scores[valid]
    pd_boxes = conf_boxes[valid]
    pd_labels = conf_labels[valid]
    
    return pd_boxes, pd_labels


def clean_heatmap(heatmap,mode=1):
    '''
    Normalize raw heatmap such that
    - the entries are all nonnegative
    - the entries sum up to 1.0
    '''
    if mode == 1:
        min_val = np.min(heatmap)
        heatmap = heatmap - min_val # make sure heatmap is always positive
        med_val = np.median(heatmap) # take median
        heatmap[heatmap < med_val] = 0 # get rid of all values below median

    elif mode == 2:
        min_val = np.min(heatmap)
        if min_val < 0:
            heatmap = heatmap - min_val

    else:
        raise RuntimeError('Unknown mode for cleaning heatmap')
    heatmap = heatmap / np.sum(heatmap)
    return heatmap

def topk_points(heatmap,k):
    '''
    Return the top k most likely keypoint detections in the heatmap
    xy: xy coordinates of the keypoints
    vk: values of the top k probabilities (re-normalized to sum up to 1.0)
    '''
    r, c = np.unravel_index(
            np.flip(np.argsort(heatmap.ravel())), heatmap.shape)
    v    = heatmap[r,c]
    rk = r[:k]
    ck = c[:k]
    vk = v[:k]
    vk = vk / np.sum(vk)
    # offset the coordinates to the center
    # For example (0,0) pixel has coordinates (0.5,0.5)
    ck = ck + 0.5 
    rk = rk + 0.5
    xy = np.stack((ck,rk),axis=1)
    return xy, vk


def conformity_score(kpt,heatmap,type="ball"):
    '''
    Given a keypoint location on a 2D image, and 
    a heatmap prediction of the keypoint location,
    compute the nonconformility score
    :param
    kpt: (2,) numpy array
    heatmap: (H,W) numpy array
    type: choice of the conformity function
    :return
    conformity score
    '''

    heatmap = clean_heatmap(heatmap,mode=1)
       
    if type == "ball":
        r, c = np.unravel_index(
            np.argmax(heatmap.ravel()),heatmap.shape)
        maxp = heatmap[r,c]
        # note here kpt loc (x,y), x corresponds to column, y corresponds to row!!!
        r += 0.5
        c += 0.5
        dist = np.linalg.norm( kpt - np.array([c,r]) )
        return dist * maxp

    elif type == "ellipse":
        xy, v = topk_points(heatmap,K)
        wkpt   = v @ xy
        diff   = xy - wkpt
        sigma  = diff.T @ np.diag(v) @ diff
        sigmainv = np.linalg.inv(sigma)
        return (kpt - wkpt) @ sigmainv @ (kpt-wkpt)

    else:
        raise RuntimeError('Unknown score type.')


def icp(heatmap,q,type="ball"):
    '''
    Given a heatmap and a quantile, output the inductive prediction set
    :param
    heatmap: numpy array H x W
    q: scalar quantitle
    type: choice of conformity function
    '''

    heatmap = clean_heatmap(heatmap,mode=1)

    if type == "ball":
        r, c = np.unravel_index(
            np.argmax(heatmap.ravel()),heatmap.shape)
        maxp = heatmap[r,c]
        c += 0.5
        r += 0.5
        return np.array([c,r]), q / maxp # return center and radius

    elif type == "ellipse":
        xy, v = topk_points(heatmap,K)
        wkpt   = v @ xy
        diff   = xy - wkpt
        sigma  = diff.T @ np.diag(v) @ diff
        sigmainv = np.linalg.inv(sigma)
        return wkpt, sigmainv / q # return center and information matrix

    else:
        raise RuntimeError('Unknown score type.')
    

def draw_icp_ball(img,heatmaps,kpt_gt,pred_set,fname=None,show=False,heatmaponly=False):
    linewidth = 2
    pointsize = 2
    height = 20
    subplot_gap = 0.05
    num_kpts = len(pred_set)
    colors  = cm.Set2(np.linspace(0, 1, num_kpts))

    fig, axes = plt.subplots(1,num_kpts+1,figsize=(2*height,2*height))
    fig.subplots_adjust(wspace=subplot_gap)

    for i in range(num_kpts):
        heatmap = np.squeeze(heatmaps[i,:,:])
        heatmap = clean_heatmap(heatmap)
        
        axes[i].imshow(img)
        axes[i].imshow(heatmap,alpha=alpha)
        if not heatmaponly:
            center, radius = pred_set[i]
            circ = plt.Circle(center,radius,color=colors[i],fill=True,linewidth=linewidth,alpha=0.5)
            axes[i].add_patch(circ)
            circ_b = plt.Circle(center,radius,color=colors[i],fill=False,linewidth=linewidth)
            axes[i].add_patch(circ_b)
            # point = plt.Circle((kpt_gt[i,0],kpt_gt[i,1]),pointsize,color=colors[i])
            point = plt.Rectangle([kpt_gt[i,0]-pointsize/2,kpt_gt[i,1]-pointsize/2],pointsize,pointsize,color=colors[i])
            axes[i].add_patch(point)
        axes[i].xaxis.set_visible(False)
        axes[i].yaxis.set_visible(False)

    axes[-1].imshow(img)
    for i in range(num_kpts):
        center, radius = pred_set[i]
        circ = plt.Circle(center,radius,color=colors[i],fill=True,linewidth=linewidth,alpha=0.5)
        axes[-1].add_patch(circ)
        circ_b = plt.Circle(center,radius,color=colors[i],fill=False,linewidth=linewidth)
        axes[-1].add_patch(circ_b)
        # point = plt.Circle((kpt_gt[i,0],kpt_gt[i,1]),pointsize,color=colors[i])
        point = plt.Rectangle([kpt_gt[i,0]-pointsize/2,kpt_gt[i,1]-pointsize/2],pointsize,pointsize,color=colors[i])
        axes[-1].add_patch(point)
    axes[-1].xaxis.set_visible(False)
    axes[-1].yaxis.set_visible(False)

    if fname is not None:
        plt.savefig(fname,bbox_inches='tight')
    if show:
        plt.show()
    
    return fig 


def angle_length_ellipse(A):
    '''
    Given an ellipse x' * A * x <= 1
    return a, b, and angle
    angle is the angle rotating from x to y (anti-clockwise)
    '''
    v, V = np.linalg.eig(A)
    idx  = np.argsort(v)
    v = v[idx] # ascending order v[0] <= ... <= v[-1]
    V = V[:,idx]

    ab = np.sqrt(1.0 / v)
    a  = ab[0]
    b  = ab[-1]
    assert a >= b, "semi-axes lengths wrong."

    Vl = V[:,0] # long axis direction
    angle = np.arctan2(Vl[-1],Vl[0]) / np.pi * 180.0

    return a, b, angle


def draw_icp_ellipse(img,heatmaps,kpt_gt,pred_set,fname=None,show=False):
    linewidth = 2
    pointsize = 2
    height = 20
    subplot_gap = 0.05
    num_kpts = len(pred_set)
    colors  = cm.Set2(np.linspace(0, 1, num_kpts))

    fig, axes = plt.subplots(1,num_kpts+1,figsize=(2*height,2*height))
    fig.subplots_adjust(wspace=subplot_gap)

    for i in range(num_kpts):
        heatmap = np.squeeze(heatmaps[i,:,:])
        heatmap = clean_heatmap(heatmap)
        
        axes[i].imshow(img)
        axes[i].imshow(heatmap,alpha=alpha)
        center, lam = pred_set[i]
        a, b, angle = angle_length_ellipse(lam)
        ellipse = matplotlib.patches.Ellipse(center,2*a,2*b,angle=angle,color=colors[i],fill=True,linewidth=linewidth,alpha=0.5)
        axes[i].add_patch(ellipse)
        ellipse_b = matplotlib.patches.Ellipse(center,2*a,2*b,angle=angle,color=colors[i],fill=False,linewidth=linewidth)
        axes[i].add_patch(ellipse_b)
        point = plt.Rectangle([kpt_gt[i,0]-pointsize/2,kpt_gt[i,1]-pointsize/2],pointsize,pointsize,color=colors[i])
        axes[i].add_patch(point)
        axes[i].xaxis.set_visible(False)
        axes[i].yaxis.set_visible(False)

    axes[-1].imshow(img)
    for i in range(num_kpts):
        center, lam = pred_set[i]
        a, b, angle = angle_length_ellipse(lam)
        ellipse = matplotlib.patches.Ellipse(center,2*a,2*b,angle=angle,color=colors[i],fill=True,linewidth=linewidth,alpha=0.5)
        axes[-1].add_patch(ellipse)
        ellipse_b = matplotlib.patches.Ellipse(center,2*a,2*b,angle=angle,color=colors[i],fill=False,linewidth=linewidth)
        axes[-1].add_patch(ellipse_b)
        point = plt.Rectangle([kpt_gt[i,0]-pointsize/2,kpt_gt[i,1]-pointsize/2],pointsize,pointsize,color=colors[i])
        axes[-1].add_patch(point)
    axes[-1].xaxis.set_visible(False)
    axes[-1].yaxis.set_visible(False)

    if fname is not None:
        plt.savefig(fname,bbox_inches='tight')
    if show:
        plt.show()
    
    return fig
