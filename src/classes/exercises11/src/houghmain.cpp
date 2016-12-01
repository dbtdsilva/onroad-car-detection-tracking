//
//  main.cpp
//  TestHough
//
//  Created by Saburo Okita on 27/04/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

#include "Hough.h"

using namespace cv;
using namespace std;

static Point accumIndex(-1, -1);
static void onMouse( int event, int x, int y, int, void * data );

int main(int argc, const char * argv[]) {
    namedWindow( "" );
    moveWindow("", 0, 0);
    
    Mat image = imread(argv[1]);
    resize( image, image, Size(), 1.0, 1.0 );
    
    const double resize_factor_x = 1;
    const double resize_factor_y = 1;
    
    /* Find the edges of the image */
    Mat gray, edges;
    cvtColor( image, gray, CV_BGR2GRAY );
    
    
    bool show_canny = false;
    int threshold = 0;
    int ratio = 3;
    int canny_thresh = 100;
    while( true ) {
        Canny( gray, edges, canny_thresh, canny_thresh * ratio, 3 );
        Hough hough;
        hough.init( edges );   
        /* Get the size of accum matrix in advance, for the mouse call back purpose  */
        Mat accum = hough.getAccumulationMatrix();
        resize( accum, accum, Size(), 2.0 * resize_factor_x, 0.5 * resize_factor_y );

        Rect region( image.cols, 0, accum.cols, accum.rows );
        setMouseCallback( "", onMouse, static_cast<void*>( &region ) );
        /* Final output matrix, will be combination of the original image and the accumulation matrix */
        Mat appended( MAX( accum.rows, image.rows ), image.cols + accum.cols, CV_8UC3, Scalar(0, 0, 0) );

        char str[255];   
        createTrackbar( "Hough threshold", "", &threshold, 1000 ); 
        createTrackbar( "First threshold", "", &canny_thresh, 400 ); 
        createTrackbar( "Canny ratio (Second threshold = first * ratio)", "", &ratio, 10 ); 
        Mat temp = image.clone();
        //if( show_canny )
        //    temp += edges;
        
        /* Try to visualize the accumulation matrix */
        accum = hough.getAccumulationMatrix( threshold );
        accum.convertTo( accum, CV_8UC1 );
        equalizeHist( accum, accum );
        
        /* Apply colormap for better representation of the accum matrix */
        applyColorMap( accum, accum, cv::COLORMAP_OCEAN );
        resize( accum, accum, Size(), 2.0 * resize_factor_x, 0.5 * resize_factor_y);
        
        /* Draw the lines based on threshold */
        vector<pair<Point, Point>> lines = hough.getLines( threshold );
        for( pair<Point, Point> point_pair : lines )
            line( temp, point_pair.first, point_pair.second, CV_RGB(255, 0, 0), 1 );
    
        
        /* Draw lines based on cursor position */
        if(accumIndex.x != -1 && accumIndex.y != -1 ) {
            pair<Point, Point> point_pair = hough.getLine( accumIndex.y / resize_factor_y, accumIndex.x / resize_factor_x);
            line( temp, point_pair.first, point_pair.second, CV_RGB(0, 255, 0), 1 );
        }
        
        /* Copy everything to output matrix */
        appended = Scalar::all(0);
        temp.copyTo ( Mat(appended, Rect(0, 0, temp.cols, temp.rows)) );
        accum.copyTo( Mat(appended, Rect(temp.cols, 0, accum.cols, accum.rows))  );
        
        
        /* Output some text */
        putText( appended, "[Q] to quit", cvPoint( 10, 80 ), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);
        sprintf( str, "Threshold: %d", threshold );
        putText( appended, str, cvPoint( 10, 100 ), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);
        sprintf( str, "Rho: %d   Theta: %d", accumIndex.y - accum.rows / 2, accumIndex.x );
        putText( appended, str, cvPoint( 10, 120 ), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);

        
        imshow( "", appended );
        imshow( "Canny", edges );
        char key = waitKey(10);
        if( key == 'q' )
            break;
    }
    
    return 0;
}


/**
 * Mouse callback, to show the line based on which part of accumulation matrix the cursor is.
 */
static void onMouse( int event, int x, int y, int, void * data ) {
    Rect *region = (Rect*) data;
    Point point( x, y );
    
    if( (*region).contains( point ) ) {
        accumIndex.x = (point.x - region->x) / 2.0;
        accumIndex.y = (point.y - region->y) * 2.0;
    }
    else {
        accumIndex.x = -1;
        accumIndex.y = -1;
    }
}