#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "panoramic_utils.h"
#include "PanoramicImage.h"
#include <opencv2/xfeatures2d.hpp>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    cv::String dir = "/home/seleucio/Documenti/dolomites/*.png";
    PanoramicImage image = PanoramicImage(dir, 54);
    Mat panoramic = image.createPanoramicImage(true);
    imshow("Panoramic", panoramic);
    cv::waitKey(0);
    return 0;
}
