#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/visualization/cloud_viewer.h>

int  main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  (*cloud).width    = 50;
  (*cloud).height   = 50;
  (*cloud).is_dense = false;
  (*cloud).points.resize ((*cloud).width * (*cloud).height);
 
  for (size_t i = 0; i < (*cloud).points.size (); ++i)
  {
    (*cloud).points[i].x = 10 * rand () / (RAND_MAX + 1.0f);
    (*cloud).points[i].y = 10 * rand () / (RAND_MAX + 1.0f);
    (*cloud).points[i].z = 10 * rand () / (RAND_MAX + 1.0f);
  }

  
  pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
  viewer.showCloud(cloud);
  
  while (!viewer.wasStopped())
    {
    }
 
  return (0);
}