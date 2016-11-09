#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

int main(int argc, char **argv) {
    if (argc < 2) {
        cout << "Usage: ./viewCloudFile filename [true - to enable grid]" << endl;
        cout << "Expected filename from the pcd file." << endl;
        return -1;
    }
    bool grid = false;
    if (argc > 2 && string(argv[2]) == "true") {
        grid = true;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (argv[1], *cloud) == -1) {
        PCL_ERROR ("Couldn't read file filt_office1.pcd \n");
        return -2;
    }

    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");



    if (grid) {
        cout << "Number of points before: " << cloud->points.size() << endl;
        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
        voxel_grid.setInputCloud(cloud);
        voxel_grid.setLeafSize (0.05, 0.05, 0.05);
        voxel_grid.filter(*cloud);
        cout << "Number of points after: " << cloud->points.size() << endl;
    }
    viewer.showCloud(cloud);

    while (!viewer.wasStopped()) {
    }

    return (0);
}