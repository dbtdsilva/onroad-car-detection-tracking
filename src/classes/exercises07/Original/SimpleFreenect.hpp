#ifndef SIMPLE_FREENECT_HPP
#define SIMPLE_FREENECT_HPP

/**
 * \file
 * Define SimpleFreenect class.
 */

#include <opencv2/opencv.hpp>
//#include <libfreenect/libfreenect.h>
//#include <libfreenect.h>
#include <pthread.h>

//using FLANN from OpenCVnamespace cv;

/**
 * C++ wrapper libfreenect in an OpenCV friendly way.
 *
 * Access to the hardware buffers is made using callback functions. Mechanisms
 * are in place to ensure mutual exclusion in data access.
 */

class SimpleFreenect
{
	public:
		/** Constructor */
		SimpleFreenect();
		
		/** Destructor */
		~SimpleFreenect();
		
		/**
		 * Start receiving video frames.
		 */
		void startVideo(void);

		/**
		 * Start receiving depth frames.
		 */
		void startDepth(void);
		
		/** 
		 * Stop receiving frames.
		 */
		void stopVideo(void);

		/** 
		 * Stop receiving depth frames.
		 */
		void stopDepth(void);
		
		/** 
		 * Try to grab the latest BGR frame. The grabbed frame can be accessed
		 * in the BGR attribute.
		 *
		 * \returns True if frame was successfully grabbed, false otherwise.
		 */
    bool grabBGR(cv::Mat &BGR);

		/**
		 * Try to grab the latest depth frame. The grabbed frame can be
		 * accessed in the depth attribute.
		 *
		 * \returns true if frame was successfully grabbed, false otherwise.
		 */
    bool grabDepth(cv::Mat &depth);

		/**
		 * Set tilt servo angle, in degrees.
		 *
		 * \param[in] angle Angle to set the servo to. 0 corresponds to
		 * horizontal.
		 */
		void setTilt(double angle);

		/**
		 * Get tilt servo angle, in degrees.
		 *
		 * \returns Servo current angle, in degrees. 0 corresponds to horizontal.
		 */
		double getTilt(void);

		/**
		 * Get accelerometers information.
		 *
		 * \param[out] ax X axis acceleration.
		 * \param[out] ay Y axis acceleration.
		 * \param[out] az Z axis acceleration.
		 */
		void getAccelerometers(double *ax, double *ay, double *az);

		/**
		 * Do not call. Internal callback function.
		 */
		void videoCallback(void *video, uint32_t timestamp);

		/**
		 * Do not call. Internal callback function.
		 */
		void depthCallback(void *depth, uint32_t timestamp);

		/**
		 * Do not call. Internal callback function.
		 */
		void processEventsCallback();

	private:
		// Internal buffers and book keeping variables
		uint8_t depthBuffer[FREENECT_DEPTH_11BIT_SIZE];
    cv::Mat depthMat;
		unsigned int depthFrameCount;
		bool newDepthFrame;
		
		uint8_t videoBuffer[FREENECT_VIDEO_RGB_SIZE];
    cv::Mat videoMat;
		unsigned int videoFrameCount;
		bool newVideoFrame;

		// libfreenect specific stuff
		freenect_context *ctx;
		freenect_device *dev;

		pthread_t thread;
		bool running;

		// Throw exception when libfreenect error occurs
		void throwOnError(int retval);
};

#endif
