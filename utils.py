from distutils.command.clean import clean
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib

K     = 100
alpha = 0.8
topk_heatmap_mode = 2

def clean_heatmap(heatmap,mode=1):
    if mode == 1:
        min_val = np.min(heatmap)
        heatmap = heatmap - min_val # make sure heatmap is always positive
        med_val = np.median(heatmap) # take median
        heatmap[heatmap < med_val] = 0 # get rid of all values below median

    elif mode == 2:
        min_val = np.min(heatmap)
        if min_val < 0:
            heatmap = heatmap - min_val

    elif mode == 3:
        heatmap = np.exp(heatmap)

    else:
        raise RuntimeError('Unknown mode for cleaning heatmap')
    heatmap = heatmap / np.sum(heatmap)
    return heatmap

def topk_points(heatmap,k):
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


def conformity_score(kpt,heatmap,type="topk",mode=1):
    '''
    Given a keypoint location on a 2D image, and 
    a heatmap prediction of the keypoint location,
    compute the conformility score
    :param
    kpt: (2,) numpy array
    heatmap: (H,W) numpy array
    type: choice of the conformity function
    mode: 1 rounds the kpt location and take probability of one pixel
    2 takes the average probability of four surrounding pixels
    :return
    conformity score
    '''
    if type == "topk":
        heatmap = clean_heatmap(heatmap,mode=topk_heatmap_mode)
    else:
        heatmap = clean_heatmap(heatmap,mode=1)

    if type == "topk":
        if mode == 1:
            kpt_int = np.int64( np.floor(kpt) )
            y       = heatmap[kpt_int[1],kpt_int[0]] # note here kpt loc (x,y), x corresponds to column, y corresponds to row!!!

        elif mode == 2:
            kpt_int = np.int64( np.floor(kpt) )
            y1      = heatmap[kpt_int[1],kpt_int[0]]
            y2      = heatmap[kpt_int[1]+1,kpt_int[0]]
            y3      = heatmap[kpt_int[1],kpt_int[0]+1]
            y4      = heatmap[kpt_int[1]+1,kpt_int[0]+1]
            y       = 0.25 * (y1 + y2 + y3 + y4)
        else:
            raise RuntimeError('Unknown mode for computing conformity score.')
        heatmap[heatmap < y] = 0
        q = np.sum(heatmap)
        # print(f'q={q},y={y}')
        return q

    elif type == "radius-maxp":
        r, c = np.unravel_index(
            np.argmax(heatmap.ravel()),heatmap.shape)
        maxp = heatmap[r,c]
        # note here kpt loc (x,y), x corresponds to column, y corresponds to row!!!
        r += 0.5
        c += 0.5
        dist = np.linalg.norm( kpt - np.array([c,r]) )
        return dist * maxp

    elif type == "radius-variance":
        nr, nc = heatmap.shape
        rr, cc = np.meshgrid(np.arange(0,nr),np.arange(0,nc),indexing='ij')
        v      = np.reshape(heatmap,(nr*nc,),order='F')
        rr     = np.reshape(rr,(nr*nc,),order='F')
        cc     = np.reshape(cc,(nr*nc,),order='F')
        rr += 0.5
        cc += 0.5
        xy     = np.stack((cc,rr),axis=1)
        wkpt   = v @ xy
        diff   = xy - wkpt
        wr     = np.sqrt( np.sum(diff**2,axis=1) @ v )
        return np.linalg.norm(kpt - wkpt) / wr

    elif type == "radius-cov":
        nr, nc = heatmap.shape
        rr, cc = np.meshgrid(np.arange(0,nr),np.arange(0,nc),indexing='ij')
        v      = np.reshape(heatmap,(nr*nc,),order='F')
        rr     = np.reshape(rr,(nr*nc,),order='F')
        cc     = np.reshape(cc,(nr*nc,),order='F')
        rr += 0.5
        cc += 0.5
        xy     = np.stack((cc,rr),axis=1)
        wkpt   = v @ xy
        diff   = xy - wkpt
        sigma  = diff.T @ np.diag(v) @ diff
        sigmainv = np.linalg.inv(sigma)
        return (kpt - wkpt) @ sigmainv @ (kpt-wkpt)

    elif type == "radius-var-topk":
        xy, v = topk_points(heatmap,K)
        wkpt   = v @ xy
        diff   = xy - wkpt
        wr     = np.sqrt( np.sum(diff**2,axis=1) @ v )
        return np.linalg.norm(kpt - wkpt) / wr

    elif type == "radius-cov-topk":
        xy, v = topk_points(heatmap,K)
        wkpt   = v @ xy
        diff   = xy - wkpt
        sigma  = diff.T @ np.diag(v) @ diff
        sigmainv = np.linalg.inv(sigma)
        return (kpt - wkpt) @ sigmainv @ (kpt-wkpt)

    else:
        raise RuntimeError('Unknown score type.')


def icp(heatmap,q,type="topk"):
    '''
    Given a heatmap and a quantile, output the inductive prediction set
    :param
    heatmap: numpy array H x W
    q: scalar quantitle
    type: choice of conformity function
    :return
    if "topk": numpy array C x 2, where each row is a candidate location
    if "radius-maxp": tuple (center, radius) that defines a circle 
    '''
    if type == "topk":
        heatmap = clean_heatmap(heatmap,mode=topk_heatmap_mode)
    else:
        heatmap = clean_heatmap(heatmap,mode=1)

    if type == "topk":
        # obtain row and column indices with values sorted in descending order
        r, c = np.unravel_index(
            np.flip(np.argsort(heatmap.ravel())), heatmap.shape)
        
        total_prob = 0
        for i in range(np.size(heatmap)):
            total_prob = total_prob + heatmap[r[i],c[i]]
            if total_prob > q:
                break

        # print(f'q={q},i={i}')
        
        rc = np.stack((c,r),axis=1)
        return rc[:i,:]

    elif type == "radius-maxp":
        r, c = np.unravel_index(
            np.argmax(heatmap.ravel()),heatmap.shape)
        maxp = heatmap[r,c]
        c += 0.5
        r += 0.5
        return np.array([c,r]), q / maxp

    elif type == "radius-variance":
        nr, nc = heatmap.shape
        rr, cc = np.meshgrid(np.arange(0,nr),np.arange(0,nc),indexing='ij')
        v      = np.reshape(heatmap,(nr*nc,),order='F')
        rr     = np.reshape(rr,(nr*nc,),order='F')
        cc     = np.reshape(cc,(nr*nc,),order='F')
        rr += 0.5
        cc += 0.5
        xy     = np.stack((cc,rr),axis=1)
        wkpt   = v @ xy
        diff   = xy - wkpt
        wr     = np.sqrt( np.sum(diff**2,axis=1) @ v )
        return wkpt, wr*q

    elif type == "radius-var-topk":
        xy, v = topk_points(heatmap,K)
        wkpt   = v @ xy
        diff   = xy - wkpt
        wr     = np.sqrt( np.sum(diff**2,axis=1) @ v )
        return wkpt, wr*q

    elif type == "radius-cov":
        nr, nc = heatmap.shape
        rr, cc = np.meshgrid(np.arange(0,nr),np.arange(0,nc),indexing='ij')
        v      = np.reshape(heatmap,(nr*nc,),order='F')
        rr     = np.reshape(rr,(nr*nc,),order='F')
        cc     = np.reshape(cc,(nr*nc,),order='F')
        rr += 0.5
        cc += 0.5
        xy     = np.stack((cc,rr),axis=1)
        wkpt   = v @ xy
        diff   = xy - wkpt
        sigma  = diff.T @ np.diag(v) @ diff
        sigmainv = np.linalg.inv(sigma)
        return wkpt, sigmainv / q # center, and PD matrix of the ellipse 

    elif type == "radius-cov-topk":
        xy, v = topk_points(heatmap,K)
        wkpt   = v @ xy
        diff   = xy - wkpt
        sigma  = diff.T @ np.diag(v) @ diff
        sigmainv = np.linalg.inv(sigma)
        return wkpt, sigmainv / q

    else:
        raise RuntimeError('Unknown score type.')

def is_kpt_in_set(p,set):
    '''
    check if a point p is in the set
    :param
    p: (d,) numpy array
    set: (N,d) numpy array
    both p and set should have integer values
    :return
    True or false
    '''
    p = np.floor(p)

    dist = np.linalg.norm(set - p,axis=1)
    min_dist = np.min(dist)
    return (min_dist < 1e-6)
    

def draw_icp_circle(img,heatmaps,kpt_gt,pred_set,fname=None,show=False,heatmaponly=False):
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

def draw_icp_topk(img,heatmaps,kpt_gt,pred_set,fname=None,show=False):
    linewidth = 2
    pointsize = 2
    height = 20
    subplot_gap = 0.05
    num_kpts = len(pred_set)
    colors  = cm.Set2(np.linspace(0, 1, num_kpts))

    fig, axes = plt.subplots(1,num_kpts+1,figsize=(height,height))
    fig.subplots_adjust(wspace=subplot_gap)

    for i in range(num_kpts):
        heatmap = np.squeeze(heatmaps[i,:,:])
        heatmap = clean_heatmap(heatmap)
        
        axes[i].imshow(img)
        axes[i].imshow(heatmap,alpha=alpha)
        topk = pred_set[i]
        for j in range(topk.shape[0]):
            topk_j = topk[j,:]
            point = plt.Rectangle([topk_j[0]-1/2,topk_j[1]-1/2],1,1,color=colors[i])
            axes[i].add_patch(point)

        point = plt.Rectangle([kpt_gt[i,0]-pointsize/2,kpt_gt[i,1]-pointsize/2],pointsize,pointsize,color=colors[i])
        axes[i].add_patch(point)
        axes[i].xaxis.set_visible(False)
        axes[i].yaxis.set_visible(False)

    axes[-1].imshow(img)
    for i in range(num_kpts):
        topk = pred_set[i]
        for j in range(topk.shape[0]):
            topk_j = topk[j,:]
            point = plt.Rectangle([topk_j[0]-1/2,topk_j[1]-1/2],1,1,color=colors[i])
            axes[-1].add_patch(point)

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
