#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

void cornernessHarris()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // visualize results
    string windowName = "Harris Corner Detector Response Matrix";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, dst_norm_scaled);
    cv::waitKey(0);

    // TODO: Your task is to locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`.
    std::vector<cv::KeyPoint> keyPoints;
    for (int r = 0; r < dst_norm.rows; ++r)
    {
        for (int c = 0; c < dst_norm.cols; ++c)
        {
            int resp = static_cast<int>(dst_norm.at<float>(r, c));
            if (resp >= minResponse)
            {
                // Need to consider this point as a new keypoint
                cv::KeyPoint pt(c, r, 2 * apertureSize, -1.0, resp);

                // Check if there is higher response keypoint already in the vector
                bool addPt = true;
                for (std::vector<cv::KeyPoint>::iterator it = keyPoints.begin(); it != keyPoints.end(); ++it)
                {
                    // Does the new keypoint overlap with the point in the list
                    double overlap = cv::KeyPoint::overlap(pt, *it);
                    if (overlap > 0.0)
                    {
                        //The points do overlap. Check the response
                        if (pt.response > (*it).response)
                        {
                            // Replace the existing point with this one
                            *it = pt;
                            addPt = false;
                            break;
                        }
                    }
                }  

                if (addPt)
                {
                    keyPoints.push_back(pt);
                }             
            }
        }
    }

        // visualize results
    windowName = "Harris Corner Detector Results";
    cv::namedWindow(windowName, 5);
    cv::Mat visImage = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, keyPoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);


}

int main()
{
    cornernessHarris();
}