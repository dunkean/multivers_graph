
import trimesh
import numpy as np
import random
# import scipy.spatial.transform as t
import transforms3d
# from transforms3d.euler import euler2mat, mat2euler


def R():
    return transforms3d.euler.euler2mat(0, 0, 0, 'sxyz')

def T(h,t):
    return transforms3d.affines.compose([0+t[0],h/2+t[1],0+t[2]], R(), [1.0,1.0,1.0])

## height_ref = 1.8, TODO auto resize for other heights
def sub_mesh_cube(mesh, size, t=(0,0,0)):    
    cube = trimesh.creation.box(size)
    cube.apply_transform(T(size[1],t))
    return trimesh.boolean.intersection([mesh, cube])

def sub_mesh_sphere(mesh, radius, t=[0,0,0]):
    sphere = trimesh.creation.icosphere(1, radius)
    sphere.apply_transform(T(radius,t))
    return trimesh.boolean.intersection([mesh, sphere])

def norm_mesh(mesh, height):
    bds = mesh.bounds
    T = [0, -bds[0,1], 0]
    h = bds[1,1] - bds[0,1]
    yscale = 1.8 / h
    Z = [yscale,yscale,yscale] # zooms
    A = transforms3d.affines.compose(T, R(), Z)
    mesh.apply_transform(A)

def mesh_cube(size, t=[0,0,0]):    
    cube = trimesh.creation.box(size)
    cube.apply_transform(T(size[1],t))
    return cube

def mesh_sphere(radius, t=[0,0,0]):    
    cube = trimesh.creation.icosphere(1, radius)
    cube.apply_transform(T(radius,t))
    return cube

def _hash(a):
        return hash(a.tobytes())

# man = trimesh.load("woman.obj")
man = trimesh.load("man.obj")
norm_mesh(man, 1.8)
print("Model loaded and normalized")
man_vox = man.voxelized(pitch=0.03)
envelope_points = man_vox.points.copy()
# print(len(envelope_points), envelope_points.shape)
man_vox.fill()
# all_points = man_vox.points
# print(len(all_points), all_points.shape)

# inner_points = list(set(all_points) - set(envelope_points))

# print("Voxels generated")

# head = sub_mesh_sphere(man,0.15, (0,1.6,0.03))
# print("Intersection done")

# head = sub_mesh_sphere(man,0.15, (0,1.6,0.03))
# lfeet = sub_mesh_cube(man, (0.3,0.13,0.4), (-0.2,0,0))
# rfeet = sub_mesh_cube(man, (0.3,0.13,0.4), (0.2,0,0))
# lleg = sub_mesh_cube(man, (0.3,0.7,0.3), (0.16,0.1,0))
# rleg = sub_mesh_cube(man, (0.3,0.7,0.3), (-0.16,0.1,0))
# abdomen = sub_mesh_cube(man, (0.35,0.62,0.4), (0,0.9,0))
# thorax = sub_mesh_cube(man, (0.35,0.30,0.4), (0,1.22,0))
# belly = sub_mesh_cube(man, (0.35,0.32,0.4), (0,0.9,0))
# sex = sub_mesh_sphere(man, 0.15, (0,0.81,0))
# neck = sub_mesh_cube(man, (0.15,0.1,0.15), (0,1.5,0))
# brain = sub_mesh_sphere(man, 0.095, (0,1.68,0.01))


# rleg = sub_mesh_cube((0.3,0.7,0.3), -0.16, 0.1)
# cube = cube_mesh((0.15,0.1,0.15), (0,1.5,0))
# cube = sphere_mesh(0.095, (0,1.68,0.01))
# head_box = mesh_sphere(0.15, (0,1.6,0.03))


# surface_mesh_pts = man.vertices
# surface_sample_pts = man.sample(5000)
# contained = man.contains(envelope_points)
# surface_vox_out_pts = [envelope_points[i] for i in range(len(contained)) if contained[i] == False ]
# surface_vox_in_pts = [envelope_points[i] for i in range(len(contained)) if contained[i] == True ]


# in_points = [man_vox.points[i] for i in range(len(contained)) if contained[i] == True ]

volume_sample_pts = trimesh.sample.volume_mesh(man, 5000)
# print(envelope_points.shape, man_vox.points.shape)
env_hash = [_hash(a) for a in envelope_points]
volume_vox_pts = [p for p in man_vox.points if _hash(p) not in env_hash ]

# print(man_vox.points[4000] in envelope_points)
# print(volume_vox_pts)

# cloud_in = trimesh.points.PointCloud(man.sample(5000))
cloud_in = trimesh.points.PointCloud(volume_vox_pts)
cloud_out = trimesh.points.PointCloud(man.vertices)
# cloud_in = trimesh.points.PointCloud(inner_points)
# man_vox = cloud_original.voxelized(pitch=0.02)
scene = trimesh.Scene([cloud_in, cloud_out])
print("Scene build")
# scene = trimesh.Scene([man,cloud_original])

scene.show()

## scale and translate the model

# bds = mesh.bounds
# mesh.show()

#A chaque fois - center, outer, inner /if area: left-right
# areas:
# hands, head, arms, knees
# points:
# eyes, mouth, hears, nose, butt, tits, biceps, heart, lungs, liver, plexus

# fingers ?


# def add_nexus:



## extract parts
# cube = trimesh.creation.box((0.3,0.3,0.3))
# R = transforms3d.euler.euler2mat(0, 0, 0, 'sxyz')
# A = transforms3d.affines.compose([0,0,0], R, [1.0,1.0,1.0])
# cube.apply_transform(A)

# c = trimesh.boolean.intersection([mesh, cube])




# Rx = trimesh.rotation_matrix(1.6, [1,0,0])
# rot = t.Rotation.from_euler("xyz", [90, 0, 0], degrees=True).as_matrix()
# mat = np.zeros((4,4))
# print(mat)
# mat[0:3,0:3] = rot
# mat[3,:]= [0,0,0,1]
# print(mat)
# # print(rot)
# mesh.apply_transform(mat)
# voxels = mesh.voxelized(pitch=0.1)
# voxels = voxels.fill()
# voxels.show()
# print(voxels.points.shape)
# points = random.choices(voxels.points, k=200)
# cloud_original = trimesh.points.PointCloud(points)


# mesh.show()


###LOAD MODEL
### NORMALIZE MODEL
### SAVE MODEL


# import pymesh
# mesh = pymesh.load_mesh("men.obj")


# import open3d as o3d

# mesh = o3d.io.read_triangle_mesh("man.obj")
# voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=1)
# o3d.visualization.draw_geometries([voxel_grid])




# import trimesh 
# import numpy as np

# # load a large- ish PLY model with colors    
# mesh = trimesh.load('../models/cycloidal.ply')   

# we can sample the volume of Box primitives
# points = mesh.bounding_box_oriented.sample_volume(count=200)

# # find the closest point on the mesh to each random point
# (closest_points,
#  distances,
#  triangle_id) = mesh.nearest.on_surface(points)
# print('Distance from point to surface of mesh:\n{}'.format(distances))

# # Distance from point to surface of mesh:
# # [0.15612315 0.03552731 0.02265191 0.0148177  0.01032642 0.00772476
# #  0.05167993 0.00416806 0.0488558  0.03329107]

# # create a PointCloud object out of each (n,3) list of points
# cloud_original = trimesh.points.PointCloud(points)
# # cloud_close    = trimesh.points.PointCloud(closest_points)

# # # create a unique color for each point
# cloud_colors = np.array([trimesh.visual.random_color() for i in points])
# print(cloud_colors)

# # # set the colors on the random point and its nearest point to be the same
# cloud_original.vertices_color = cloud_colors
# # cloud_close.vertices_color    = cloud_colors

# # create a scene containing the mesh and two sets of points
# scene = trimesh.Scene([cloud_original])

# # show the scene wusing 
# scene.show()



# def get_sphere_distribution(n, dmin, Ls, maxiter=1e4, allow_wall=True):
#     """Get random points in a box with given dimensions and minimum separation.
    
#     Parameters:
      
#     - n: number of points
#     - dmin: minimum distance
#     - Ls: dimensions of box, shape (3,) array 
#     - maxiter: maximum number of iterations.
#     - allow_wall: whether to allow points on wall; 
#        (if False: points need to keep distance dmin/2 from the walls.)
        
#     Return:
        
#     - ps: array (n, 3) of point positions, 
#       with 0 <= ps[:, i] < Ls[i]
#     - n_iter: number of iterations
#     - dratio: average nearest-neighbor distance, divided by dmin.
    
#     Note: with a fill density (sphere volume divided by box volume) above about
#     0.53, it takes very long. (Random close-packed spheres have a fill density
#     of 0.64).
    
#     Author: Han-Kwang Nienhuys (2020)
#     Copying: BSD, GPL, LGPL, CC-BY, CC-BY-SA
#     See Stackoverflow: https://stackoverflow.com/a/62895898/6228891 
#     """
#     Ls = np.array(Ls).reshape(3)
#     if not allow_wall:
#         Ls -= dmin
    
#     # filling factor; 0.64 is for random close-packed spheres
#     # This is an estimate because close packing is complicated near the walls.
#     # It doesn't work well for small L/dmin ratios.
#     sphere_vol = np.pi/6*dmin**3
#     box_vol = np.prod(Ls + 0.5*dmin)
#     fill_dens = n*sphere_vol/box_vol
#     if fill_dens > 0.64:
#         msg = f'Too many to fit in the volume, density {fill_dens:.3g}>0.64'
#         raise ValueError(msg)
    
#     # initial try   
#     ps = np.random.uniform(size=(n, 3)) * Ls
    
#     # distance-squared matrix (diagonal is self-distance, don't count)
#     dsq = ((ps - ps.reshape(n, 1, 3))**2).sum(axis=2)
#     dsq[np.arange(n), np.arange(n)] = np.infty

#     for iter_no in range(int(maxiter)):
#         # find points that have too close neighbors
#         close_counts = np.sum(dsq < dmin**2, axis=1)  # shape (n,)
#         n_close = np.count_nonzero(close_counts)
#         if n_close == 0:
#             break
        
#         # Move the one with the largest number of too-close neighbors
#         imv = np.argmax(close_counts)
        
#         # new positions
#         newp = np.random.uniform(size=3)*Ls
#         ps[imv]= newp
        
#         # update distance matrix
#         new_dsq_row = ((ps - newp.reshape(1, 3))**2).sum(axis=-1)
#         dsq[imv, :] = dsq[:, imv] = new_dsq_row
#         dsq[imv, imv] = np.inf
#     else:
#         raise RuntimeError(f'Failed after {iter_no+1} iterations.')

#     if not allow_wall:
#         ps += dmin/2
    
#     dratio = (np.sqrt(dsq.min(axis=1))/dmin).mean()
#     return ps, iter_no+1, dratio    