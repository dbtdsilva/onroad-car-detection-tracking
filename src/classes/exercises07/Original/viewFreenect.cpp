#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/registration/icp.h>

// OpenCV Includes
#include <cv.h>
#include <highgui.h>

void change_color(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int r, int g, int b)
{
    //Change colors
    for (int i = 0; i < (*cloud).height * (*cloud).width  ; i++)
      {
        (*cloud).points[i].r = r;
        (*cloud).points[i].g = g;
        (*cloud).points[i].b = b;
      }

}

int  main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> ("..//depth_images//filt_office1.pcd", *cloud1) == -1) //* load the file
      {
        PCL_ERROR ("Couldn't read file filt_office1.pcd \n");
        return (-1);
      }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> ("..//depth_images//filt_office2.pcd", *cloud2) == -1) //* load the file
      {
        PCL_ERROR ("Couldn't read file filt_office2.pcd \n");
        return (-1);
      }

  change_color(cloud1,255,0,0);
  change_color(cloud2,0,255,0);


   pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
   viewer.showCloud(cloud1,"cloud1");
   viewer.showCloud(cloud2,"cloud2");

   pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

   pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
   icp.setTransformationEpsilon (1e-6);
   icp.setMaxCorrespondenceDistance (0.25);
   icp.setMaximumIterations (50);

   icp.setInputSource (cloud2);
   icp.setInputTarget (cloud1);

   icp.align (*aligned_cloud);
   change_color(aligned_cloud,0,0,255);

   viewer.showCloud(aligned_cloud,"cloud_final");

   //Eigen::Matrix4f transformation = icp.getFinalTransformation ();

  
   while (!viewer.wasStopped())
    {
    }
 
  return (0);
}
