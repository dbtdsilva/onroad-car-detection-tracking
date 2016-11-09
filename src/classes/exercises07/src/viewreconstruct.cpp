#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/visualization/cloud_viewer.h>

using namespace cv;

int main(int argc, char **argv) {
    Mat image3d, image, image_colored;
    FileStorage fw("../Image3D_Reconstructed.xml", FileStorage::READ);
    fw["Image"] >> image;
    fw["Image3D"] >> image3d;
    fw["OriginalImage"] >> image_colored;
    fw.release();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    cloud->width = image3d.size().width;
    cloud->height = image3d.size().height;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    unsigned int p=0;
    for (int i = 0; i < cloud->height; i++) {
        for (int j = 0; j < cloud->width; j++) {
            if (image3d.at<cv::Vec3f>(i,j)[2] <= 0)
                continue;
            cloud->points.at(p).x = image3d.at<cv::Vec3f>(i,j)[0];
            cloud->points.at(p).y = image3d.at<cv::Vec3f>(i,j)[1];
            cloud->points.at(p).z = image3d.at<cv::Vec3f>(i,j)[2];
            cloud->points.at(p).r = image_colored.at<cv::Vec3f>(i,j).val[0];
            cloud->points.at(p).g = image_colored.at<cv::Vec3f>(i,j).val[1];
            cloud->points.at(p).b = image_colored.at<cv::Vec3f>(i,j).val[2];
            p++;
        }
    }

    imshow("RectifiedStereoImages", image);
    imshow("RectifiedStereoImagess", image_colored);
    waitKey(100);

    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    viewer.showCloud(cloud);


    while (!viewer.wasStopped()) {
    }

    return (0);
}