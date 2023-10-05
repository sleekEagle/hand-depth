import numpy as np

joint_idx={
    'root':0,
    'little': {'mcp':1, 'pip':2, 'dip':3, 'tip':4},
    'ring': {'mcp':5, 'pip':6, 'dip':7, 'tip':8},
    'middle': {'mcp':9, 'pip':10, 'dip':11, 'tip':12},
    'index': {'mcp':13, 'pip':14, 'dip':15, 'tip':16},
    'thumb': {'mcp':17, 'pip':18, 'dip':19, 'tip':20}
}

#subtract the root coordinate of each sample from the joint coordinates
# joint_coords : np.array of shape (3, n_joints, n_samples)
# root_n : index of the root joint
def joint_normalize(joint_coords,root_n=0):
    palm_centers=np.expand_dims(joint_coords[:,root_n,:],axis=1)
    palm_centers=np.repeat(palm_centers,repeats=joint_coords.shape[1] ,axis=1)
    joint_coords_norm=joint_coords-palm_centers
    return joint_coords_norm

#get hand dimentions: euclidian distances of bones of the hand

def get_euclidian_dist(a1,a2):
    return np.sqrt(np.sum(np.square(a1-a2),axis=0))


'''
get euclidian distances between consecetive hand joints
for 21 joints there will be 20 distances

items returned:
0: little_mcp - root
1: little_pip - root
2: little_dip - root
3: little_tip - root
4: ring_mcp - root
.....

kypts: np array of shape (3,n_joints,n_samples)

'''
def get_hand_dims(kypts):
    hand_dim=np.empty((0, kypts.shape[-1]))
    for key in joint_idx.keys():
        if key=='root': continue

        rel=(get_euclidian_dist(kypts[:,joint_idx[key]['mcp'],:],kypts[:,joint_idx['root'],:]))
        rel=np.expand_dims(rel,axis=0)
        hand_dim=np.append(hand_dim,rel,axis=0)
        for subkey in joint_idx[key].keys():
            if subkey=='mcp':  continue

            rel=(get_euclidian_dist(kypts[:,joint_idx[key][subkey],:],kypts[:,joint_idx[key]['mcp'],:]))
            rel=np.expand_dims(rel,axis=0)
            hand_dim=np.append(hand_dim,rel,axis=0)
    return hand_dim
