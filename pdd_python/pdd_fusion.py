import numpy as np
from numba import njit, prange, jit
from skimage import measure

class TSDFVolume():

    def __init__(self, vol_bounds, voxel_size = 0.02):

        # Define the map parameters:
        self.vol_bonds_ = vol_bounds
        self.vol_size_ = voxel_size
        self.trunc_margin_ = 5 * self.vol_size_

        self.vol_dim_ = np.ceil( (self.vol_bonds_[:,1]-self.vol_bonds_[:,0])/voxel_size ).astype(int)
        self.vol_bonds_[:,1] = self.vol_bonds_[:, 0] + self.vol_dim_ * voxel_size
        self.vol_orgin_ = self.vol_bonds_[:,0].astype(np.float32)

        print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
            self.vol_dim_[0], self.vol_dim_[1], self.vol_dim_[2],
            self.vol_dim_[0] * self.vol_dim_[1] * self.vol_dim_[2])
        )

        self.tsdf_vol_map_ = np.ones( self.vol_dim_ ).astype(np.float32)
        self.tsdf_vol_weight_ = np.ones( self.vol_dim_ ).astype(np.float32)

        # Generate the coord for each voxel:
        xv, yv, zv = np.meshgrid(
            range(self.vol_dim_[0]),
            range(self.vol_dim_[1]),
            range(self.vol_dim_[2]),
            indexing = 'ij'
        )


        self.vol_coords_ = np.concatenate([
            xv.reshape(1,-1),
            yv.reshape(1,-1),
            zv.reshape(1,-1),
        ], axis=0).astype(int).T
       #print( self.vol_coords_.shape )

    @staticmethod
    @njit(parallel=True)
    def vol2world( origin, vol_coords, vol_size ):
        #origin = origin.astype(np.float32)
        #vol_coords = vol_coords.astype(np.float32)
        vol_points = np.empty_like( vol_coords, dtype=np.float32 )
        for i in prange( vol_coords.shape[0] ):
            for j in range(3):
                vol_points[ i,j ] = origin[j] + ( vol_size * vol_coords[i,j] )
        return vol_points

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix


    @staticmethod
    @njit(parallel=True)
    def integrate_tsdf(tsdf_map, dist, pre_weight, obs_w=1):
        new_tsdf_map = np.empty_like( tsdf_map, dtype=np.float32 )
        new_weight = np.empty_like( pre_weight, dtype=np.float32 )

        for i in prange( len(tsdf_map) ):
            new_weight[i] = pre_weight[i] + obs_w
            new_tsdf_map[i] = (pre_weight[i] * tsdf_map[i] + obs_w * dist[i]) / new_weight[i]
        return new_tsdf_map, new_weight

    def integrate(self, depth_img, cam_intr, cam_pose):

        im_h , im_w  = depth_img.shape
        vol_world_points = self.vol2world( self.vol_orgin_, self.vol_coords_, self.vol_size_)
        # converet the points to the cam optical frame:
        vol_world_points_h = np.hstack( [ vol_world_points, np.ones( (len(vol_world_points),1) ) ])
        vol_cam_points_h =  ( ( np.linalg.inv(cam_pose) @ vol_world_points_h.T ).T )
        vol_cam_points = vol_cam_points_h[:,:3]

        #
        vol_cam_z = vol_cam_points[:, 2]
        #vol_pix_points_h = (cam_intr @ vol_cam_points.T).T
        #vol_pix_points =  (  ( vol_pix_points_h[:,:2] ) ) 做除法的时候取整数，不是结果取
        vol_pix_points = self.cam2pix( vol_cam_points , cam_intr )
        vol_pix_x, vol_pix_y = vol_pix_points[:,0] , vol_pix_points[:,1]

        # find the pixel inside the im_h, im_w
        valid_pix = np.logical_and( vol_pix_x>=0,
                    np.logical_and( vol_pix_x<im_w,
                    np.logical_and( vol_pix_y>=0,
                    np.logical_and( vol_pix_y<im_h,
                    vol_cam_z > 0 ) ) ) )

        depth_val = np.zeros( vol_pix_x.shape )
        depth_val[ valid_pix ] = depth_img[ vol_pix_y[valid_pix] , vol_pix_x[valid_pix] ]
        #print( depth_val[depth_val!=0] )

        # compute the TSDF distance:
        depth_diff = depth_val - vol_cam_z
        #print(depth_val)
        #print(vol_cam_z)
        valid_points = np.logical_and( depth_val >0, depth_diff >= -self.trunc_margin_ )
        dist = np.minimum(1, depth_diff/self.trunc_margin_)
        vol_valide_x = self.vol_coords_[ valid_points, 0 ]
        vol_valide_y = self.vol_coords_[ valid_points, 1 ]
        vol_valide_z = self.vol_coords_[ valid_points, 2 ]

        pre_weight = self.tsdf_vol_weight_[vol_valide_x, vol_valide_y, vol_valide_z]
        pre_tsdf_map = self.tsdf_vol_map_[vol_valide_x, vol_valide_y, vol_valide_z]
        valid_dist = dist[ valid_points ]

        if len(pre_weight)==0 or len(pre_tsdf_map)==0:
            print("img no thing")
            return
        #print(len(pre_tsdf_map), pre_tsdf_map.shape)
        new_tsdf_map, new_weight = self.integrate_tsdf(pre_tsdf_map, valid_dist, pre_weight, obs_w=1)
        #update tsdf map:
        self.tsdf_vol_weight_[vol_valide_x, vol_valide_y, vol_valide_z] = new_weight
        self.tsdf_vol_map_[vol_valide_x, vol_valide_y, vol_valide_z] = new_tsdf_map


    def get_mesh(self):

        verts, faces, norms, vals = measure.marching_cubes_lewiner(self.tsdf_vol_map_, level=0)
        verts_ind = np.round(verts).astype(int)
        verts = verts * self.vol_size_ + self.vol_orgin_

        ply_file = open("test.ply", 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n" % (verts.shape[0]))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
        ply_file.write("element face %d\n" % (faces.shape[0]))
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")

        # Write vertex list
        for i in range(verts.shape[0]):
            ply_file.write("%f %f %f %f %f %f \n" % (
                verts[i, 0], verts[i, 1], verts[i, 2],
                norms[i, 0], norms[i, 1], norms[i, 2]
            ))

        # Write face list
        for i in range(faces.shape[0]):
            ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

        ply_file.close()