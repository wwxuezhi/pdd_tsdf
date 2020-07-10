import tsdf.pdd_fusion as fusion
#import tsdf.fusion2 as fusion
import numpy as np
import cv2
import os

def get_view(img, intr, pose):
    im_h, im_w , _ = img.shape
    max_depth = np.max(img)
    # 几个角落点：
    corner_cam = np.array([
        (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[0, 0],
        (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[1, 1],
        np.array([0, max_depth, max_depth, max_depth, max_depth])
    ])
    corner_cam = corner_cam.T
    # 转到其次坐标点：
    corner_cam_h = np.hstack( [corner_cam, np.ones( ( len(corner_cam),1 ) )])
    # 转到world中去：
    corner_world = ( cam_pose @ corner_cam_h.T ).T
    #print(corner_world)
    # remove 1:
    return corner_world[: , :3]

if __name__ == "__main__":

    cam_intr = np.array( ( [277,0,160] , [0,277,120], [0,0,1] ) )
    num_image = 225
    vol_bounds = np.zeros( (3,2) )
    # preprocess data:
    for i in range( 1, num_image ):
        depth_img = cv2.imread( "/home/wei/codebuket/pdd_tsdf/data/depth%06d.png"%i )
        depth_img[  depth_img==10 ] = 0
        cam_pose =  np.loadtxt( "/home/wei/codebuket/pdd_tsdf/data/pose%06d.out"%i )

        view_in_world = get_view( depth_img, cam_intr, cam_pose)
        # format in x,y,z in columns:
        view_in_world = view_in_world.T
        vol_bounds[:, 0] = np.minimum( vol_bounds[:, 0], np.amin(view_in_world,axis=1) )
        vol_bounds[:, 1] = np.maximum( vol_bounds[:,0] , np.amax(view_in_world,axis=1) )

    print("The voxle grid map in the world range is:")
    print(vol_bounds)
    # Generate the voxel map:
    tsdf_map = fusion.TSDFVolume(vol_bounds,voxel_size=0.02)


    for i in range( 1, num_image):
        print("Fusion the frame %d/%d" %(i , num_image))
        #
        depth_img = cv2.imread("/home/wei/codebuket/pdd_tsdf/data/depth%06d.png" % i, -1)
        depth_img[ depth_img == 10 ] = 0

        cam_pose = np.loadtxt( "/home/wei/codebuket/pdd_tsdf/data/pose%06d.out"%i )

        # Intergrated into the tsdf map:
        tsdf_map.integrate( depth_img, cam_intr, cam_pose )

    tsdf_map.get_mesh()
    print("Finshied fusion, save the mesh")




