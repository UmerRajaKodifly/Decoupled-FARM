import open3d as o3d
import sys

pcd = o3d.io.read_point_cloud(sys.argv[1])

pcd_clean, inliers = pcd.remove_statistical_outlier(
    nb_neighbors=30,
    std_ratio=3.5
)

o3d.io.write_point_cloud(sys.argv[2], pcd_clean)