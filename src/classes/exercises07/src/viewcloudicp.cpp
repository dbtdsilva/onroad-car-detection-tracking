#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

int main(int argc, char **argv) {
    if (argc != 3) {
        PCL_ERROR("Expected 2 arguments: source.pcd target.pcd");
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(argv[1], *cloud1) == -1) {
        PCL_ERROR ("Couldn't read file image 1 \n");
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(argv[2], *cloud2) == -1) {
        PCL_ERROR ("Couldn't read image 2 \n");
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    icp.setTransformationEpsilon(1e-6);
    icp.setMaxCorrespondenceDistance(0.25);
    icp.setMaximumIterations(50);
    icp.setInputSource(cloud2);
    icp.setInputTarget(cloud1);
    icp.align(*aligned_cloud);

    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    viewer.showCloud(aligned_cloud);

    while (!viewer.wasStopped()) {}

    return 0;
}
