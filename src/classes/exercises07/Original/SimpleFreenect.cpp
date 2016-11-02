#include "SimpleFreenect.hpp"
#include <iostream>
#include <stdexcept>

using namespace cv;
using namespace std;

static void freenectDepthCallback(freenect_device *dev, void *depth, uint32_t timestamp)
{
	SimpleFreenect *kinect = static_cast<SimpleFreenect *>(freenect_get_user(dev));
	kinect->depthCallback(depth, timestamp);
}

static void freenectVideoCallback(freenect_device *dev, void *video, uint32_t timestamp)
{
	SimpleFreenect *kinect = static_cast<SimpleFreenect *>(freenect_get_user(dev));
	kinect->videoCallback(video, timestamp);
}

static void *pthreadCallback(void *kinect)
{
	((SimpleFreenect *)kinect)->processEventsCallback();
	return NULL;
}

SimpleFreenect::SimpleFreenect()
	: depthMat(cv::Size(640,480),CV_16UC1, (void *)depthBuffer),
	  depthFrameCount(0),
 	  newDepthFrame(false),
	  videoMat(cv::Size(640,480),CV_8UC3, (void *)videoBuffer),
	  videoFrameCount(0),
	  newVideoFrame(false),
	  running(true)
{
	// Device cannot be created more than once.
	int retval;

	retval = freenect_init(&ctx, NULL);
	throwOnError(retval);

	retval = freenect_num_devices(ctx);
	if (retval == 0) {
		cerr << "No kinect is connected.\n";
		throw "nokinect";
	} else throwOnError(retval);

	retval = freenect_open_device(ctx, &dev, 0);
	throwOnError(retval);

	// Send this pointer to be used in callbacks
	freenect_set_user(dev, this);

	// Set buffers
	freenect_set_video_buffer(dev, (void *)videoBuffer);
	freenect_set_depth_buffer(dev, (void *)depthBuffer);

	// Set capture formats, hard coded for now.
	freenect_set_video_format(dev, FREENECT_VIDEO_RGB);
	freenect_set_depth_format(dev, FREENECT_DEPTH_11BIT);

	// Set callbacks
	freenect_set_video_callback(dev, freenectVideoCallback);
	freenect_set_depth_callback(dev, freenectDepthCallback);

	// Launch thread to process usb events
	pthread_create(&thread, NULL, pthreadCallback, (void *)this);
}

SimpleFreenect::~SimpleFreenect()
{
	int retval = freenect_shutdown(ctx);
	throwOnError(retval);

	running = false;

	pthread_join(thread, NULL);

	ctx = NULL;
}

void SimpleFreenect::stopVideo(void)
{
	int retval = freenect_stop_video(dev);
	throwOnError(retval);
}

void SimpleFreenect::stopDepth(void)
{
	int retval = freenect_stop_depth(dev);
	throwOnError(retval);
}

void SimpleFreenect::startVideo(void)
{
	int retval = freenect_start_video(dev);
	throwOnError(retval);
}

void SimpleFreenect::startDepth(void)
{
	int retval = freenect_start_depth(dev);
	throwOnError(retval);
}

bool SimpleFreenect::grabBGR(Mat &image)
{
	if(newVideoFrame) {
		// Frame is not new anymore
		newVideoFrame = false;

		unsigned int thisVideoFrameCount = videoFrameCount;

		cv::cvtColor(videoMat, image, CV_RGB2BGR);

		// Check if new frame arrived while collecting
		if (videoFrameCount != thisVideoFrameCount)
			return false;
		else
			return true;
	} else
		return false;
}

bool SimpleFreenect::grabDepth(Mat &image)
{
	if(newDepthFrame) {
		// Frame is not new anymore
		newDepthFrame = false;

		unsigned int thisDepthFrameCount = depthFrameCount;
		depthMat.copyTo(image);

		// Check if new frame arrived while collecting
		if (depthFrameCount != thisDepthFrameCount)
			return false;
		else
			return true;
	} else
		return false;
}

void SimpleFreenect::setTilt(double angle)
{
	int retval = freenect_set_tilt_degs(dev, angle);
	throwOnError(retval);
}

double SimpleFreenect::getTilt(void)
{
	freenect_update_tilt_state(dev);
	return freenect_get_tilt_degs(freenect_get_tilt_state(dev));
}

void SimpleFreenect::getAccelerometers(double *ax, double *ay, double *az)
{
	freenect_update_tilt_state(dev);
	freenect_get_mks_accel(freenect_get_tilt_state(dev), ax, ay, az);
}

void SimpleFreenect::depthCallback(void *depth, uint32_t timestamp)
{
	++depthFrameCount;
	newDepthFrame = true;
}

void SimpleFreenect::videoCallback(void *video, uint32_t timestamp)
{
	++videoFrameCount;
	newVideoFrame = true;
}

void SimpleFreenect::processEventsCallback()
{
	while(running) freenect_process_events(ctx);
}

void SimpleFreenect::throwOnError(int retval)
{
	if (retval  < 0) {
		throw new runtime_error("libfreenect");
	}
}

