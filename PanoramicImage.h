#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "panoramic_utils.h"
using namespace std;
using namespace cv;
class PanoramicImage
{
private:
    Mat mergeImages(Mat left, Mat right);
    Mat findFeaturesAndDescriptions(Mat left, Mat right);
    void detectAndComputeKD(vector<Mat>&imagesOnCylinder,Ptr<ORB>& detector,vector<vector<KeyPoint>>& keypoints,vector<Mat>& descriptors);
    void detectAndComputeKD(vector<Mat>&imagesOnCylinder,Ptr<Feature2D>& detector,vector<vector<KeyPoint>>& keypoints,vector<Mat>& descriptors);
    void hardmatch(vector<vector<DMatch>>& matches, vector<Mat>& descriptors, bool advanced);
    vector<vector<DMatch>> cleaning(vector<Mat>& descriptors,vector<vector<KeyPoint>>&keypoints, vector<vector<DMatch>>& matches, bool doslopeFilter);
    void drawMatch(vector<Mat>& cylindricalImages, vector<vector<KeyPoint>>& keypoints, vector<vector<DMatch>>& good_matches);
    vector<Mat> getMaskFromHomography(vector<vector<DMatch>>& matches, vector<vector<KeyPoint>>& keypoints);
    void extractInlier(vector<vector<DMatch>>& matches, vector<Mat>& mask,vector<vector<DMatch>>& inlier_matches);
    Mat stacking(vector<Mat>& imagesOnCylinder, vector<vector<DMatch>>& inlier_matches, vector<vector<KeyPoint>>& keypoints);
    vector<vector<DMatch>> filters(vector<Mat>& descriptors, vector<vector<KeyPoint>>&keypoints, vector<vector<DMatch>>& matches, vector<vector<int>>lookup);
    void alphaBlend(Mat& img1, Mat&img2, Mat& mask, Mat& blended);
public:
    PanoramicImage(String dirname, int FoV);
    ~PanoramicImage();
    vector<Mat> load_images( const String & dir, bool showImages);
    vector<Mat> ToCylinder(int angle, string files, bool colorBGR, bool equalization);
    vector<Mat> equalizeLAB(vector<Mat> images);
    Mat createPanoramicImage(bool advanced);
    cv::String dirname;
    int angle;
    bool showImages = false;
    int nrImg;
    int img_height;
    int img_width;
};
