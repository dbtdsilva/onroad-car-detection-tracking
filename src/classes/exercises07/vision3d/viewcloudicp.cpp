#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

#define VOXEL

int main(int argc, char **argv) {
    if (argc < 3) {
        PCL_ERROR("Expected 2 arguments: source.pcd target.pcd [more_targets.pcd ..]");
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(argv[1], *source) == -1) {
        PCL_ERROR ("Couldn't read image source\n");
        return -1;
    }

#ifdef VOXEL
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
    voxel_grid.setInputCloud(source);
    voxel_grid.setLeafSize (0.01, 0.01, 0.01);
    voxel_grid.filter(*source);
#endif

    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (int i = 2; i < argc; i++) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr target(new pcl::PointCloud<pcl::PointXYZRGB>);
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(argv[i], *target) == -1) {
            PCL_ERROR ("Couldn't read file image target \n");
            return -1;
        }
#ifdef VOXEL
        pcl::VoxelGrid<pcl::PointXYZRGB> target_voxel;
        target_voxel.setInputCloud(target);
        target_voxel.setLeafSize (0.01, 0.01, 0.01);
        target_voxel.filter(*target);
#endif
        pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
        icp.setTransformationEpsilon(1e-6);
        icp.setMaxCorrespondenceDistance(0.25);
        icp.setMaximumIterations(50);
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.align(*source);

    }

    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    viewer.showCloud(source);

    while (!viewer.wasStopped()) {}

    return 0;
}
