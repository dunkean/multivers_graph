import trimesh
import numpy as np
import random
import transforms3d


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
man_vox.fill()



# surface_mesh_pts = man.vertices
# surface_sample_pts = man.sample(5000)
# contained = man.contains(envelope_points)
# surface_vox_out_pts = [envelope_points[i] for i in range(len(contained)) if contained[i] == False ]
# surface_vox_in_pts = [envelope_points[i] for i in range(len(contained)) if contained[i] == True ]

# volume_sample_pts = trimesh.sample.volume_mesh(man, 5000)
# env_hash = [_hash(a) for a in envelope_points]
# volume_vox_pts = [p for p in man_vox.points if _hash(p) not in env_hash ]
# cloud_in = trimesh.points.PointCloud(volume_vox_pts)
cloud_out = trimesh.points.PointCloud(man.vertices)

print("Scene build")

# head = mesh_sphere(0.15, (0,1.6,0.03))
# neck = mesh_cube((0.15,0.1,0.15), (0,1.5,0))
# brain = mesh_sphere(0.095, (0,1.68,0.01))
# leye = mesh_sphere(0.02, (0.035,1.69,0.11))
# reye = mesh_sphere(0.02, (-0.035,1.69,0.11))
# mouth = 
# nose = mesh_sphere(0.02, (0,1.65,0.14))
# forehead
# lear = mesh_sphere(0.03, (0.08,1.66,0.01))
# rear = mesh_sphere(0.03, (-0.08,1.66,0.01))

# lfeet = mesh_cube((0.3,0.13,0.4), (0.2,0,0))
# rfeet = mesh_cube((0.3,0.13,0.4), (-0.2,0,0))
# lleg = mesh_cube((0.3,0.7,0.3), (0.16,0.1,0))
# rleg = mesh_cube((0.3,0.7,0.3), (-0.16,0.1,0))
# lknee = mesh_sphere(0.08, (0.11,0.41,-0.05))
# rknee = mesh_sphere(0.08, (-0.11,0.41,-0.05))

# abdomen = mesh_cube((0.35,0.62,0.4), (0,0.9,0))
# thorax = mesh_cube((0.35,0.30,0.4), (0,1.22,0))
# belly = mesh_cube((0.35,0.32,0.4), (0,0.9,0))
# bellybutton = mesh_sphere(0.03, (0,1.03,0.11))
# lnipple = mesh_sphere(0.02, (0.12,1.3,0.08))
# rnipple = mesh_sphere(0.02, (-0.12,1.3,0.08))
# sex = mesh_sphere(0.15, (0,0.81,0))
# biceps, heart, lungs, liver, plexus, butt

#arms, elbows, wrist, hands, fingers


# cube = mesh_cube((0.3,0.7,0.3), (-0.16,0.1,0))
# cube = mesh_sphere(0.03, (0.08,1.66,0.01))

scene = trimesh.Scene([cloud_out, cube])
scene.show()
