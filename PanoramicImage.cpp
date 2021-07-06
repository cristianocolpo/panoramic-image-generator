#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "panoramic_utils.h"
#include "PanoramicImage.h"
#include <opencv2/xfeatures2d.hpp>
using namespace std;
using namespace cv;

PanoramicImage::PanoramicImage(cv::String dir, int FoV)
{
    dirname = dir;
    angle = FoV/2;
}

PanoramicImage::~PanoramicImage()
{
}
vector<Mat> PanoramicImage::load_images( const cv::String & dirname, bool showImages = false)
{
    vector<cv::String> files;
    vector<Mat> img_lst;
    cv::glob( dirname, files );
    
    for ( size_t i = 0; i < files.size(); ++i )
    {
        Mat img = imread( files[i] ); // load the image

        if ( img.empty() )            // invalid image, skip it.
        {
            cout << files[i] << " is invalid!" << endl;
            continue;
        }
        if ( showImages )
        {
            imshow( "image", img );
            waitKey( 1 );
        }
        img_lst.push_back( img );
    }
    nrImg = img_lst.size();
    return img_lst;
}

vector<Mat> PanoramicImage::ToCylinder(int angle, cv::String dirname, bool colorBGR, bool equalization)
{
    vector<Mat> images;
    if(equalization)
    {
        images = equalizeLAB(load_images(dirname));
    }else
    {
        images = load_images(dirname);
    }

    vector<Mat> imagesProjected;
    PanoramicUtils proj;
    for (size_t i = 0; i < nrImg; i++)
    {
        Mat projectedImage = proj.cylindricalProj(images[i],angle,colorBGR);
        imagesProjected.push_back(projectedImage);
    }
    return imagesProjected;
}

void PanoramicImage::detectAndComputeKD(vector<Mat>&imagesOnCylinder,Ptr<ORB>& detector,vector<vector<KeyPoint>>&keypoints,vector<Mat>& descriptors)
{
    for (size_t i = 0; i < nrImg; i++)
    {
        detector->detectAndCompute(imagesOnCylinder[i],noArray(),keypoints[i],descriptors[i]);
    }   
}
void PanoramicImage::detectAndComputeKD(vector<Mat>&imagesOnCylinder,Ptr<Feature2D>& detector,vector<vector<KeyPoint>>&keypoints,vector<Mat>& descriptors)
{
    for (size_t i = 0; i < nrImg; i++)
    {
        detector->detectAndCompute(imagesOnCylinder[i],noArray(),keypoints[i],descriptors[i]);
    }  
}

void PanoramicImage::hardmatch(vector<vector<DMatch>>& matches, vector<Mat>& descriptors, bool advanced)
{
    Ptr<BFMatcher> matcher;
    if(advanced)
    {
        matcher = BFMatcher::create(NORM_L2);   
    }else
    {
        matcher = BFMatcher::create(NORM_HAMMING);
    }
    for (size_t i = 0; i < nrImg -1; i++)
    {
        vector<DMatch> hardmatches;
        
        matcher->match( descriptors[i], descriptors[i+1], hardmatches);
        matches.push_back(hardmatches);
    }
}

vector<vector<DMatch>> PanoramicImage::filters(vector<Mat>& descriptors, vector<vector<KeyPoint>>&keypoints, vector<vector<DMatch>>& matches, vector<vector<int>> lookup)
{
    vector<vector<DMatch>> good_matches;
    for (int i = 0; i < nrImg-1; i++)
    {
        //here i was thinking to drop the maches with a slope different to the mean
        //but I encountered some problems 
        vector<double> slopes;
        cv::Scalar mean_s, stddev_s;
        for (int k = 0; k < lookup[i].size(); k++)
		{
            if(lookup[i][k]==1)
            {
                double deltaY =(keypoints[i][matches[i][k].queryIdx].pt.y-keypoints[i+1][matches[i][k].queryIdx].pt.y);
                double deltaX =(keypoints[i][matches[i][k].queryIdx].pt.x-keypoints[i+1][matches[i][k].queryIdx].pt.x);
                double slope = deltaY/deltaX;
                if(deltaX != 0) slopes.push_back(slope);
            }

		}
        cv::meanStdDev(slopes, mean_s, stddev_s);
        vector<DMatch> gm;
        //I will just filter the matches with distance > min_distance*ratio
        for (int k = 0; k < lookup[i].size(); k++)
        {
            if(lookup[i][k] == 1)
            {
                double deltaY =(keypoints[i][matches[i][k].queryIdx].pt.y-keypoints[i+1][matches[i][k].queryIdx].pt.y);
                double deltaX =(keypoints[i][matches[i][k].queryIdx].pt.x-keypoints[i+1][matches[i][k].queryIdx].pt.x);
                double slope = deltaY/deltaX;
                //if((mean_s[0]-0.1)<slope && slope<(mean_s[0]+0.1))
                //{
                //    if(i==0)cout <<keypoints[i][matches[i][k].queryIdx].pt.x<<" "<<keypoints[i+1][matches[i][k].queryIdx].pt.x <<" "<<slope<< endl;
                    gm.push_back(matches[i][k]);
                //}
            }

        }
        good_matches.push_back(gm);
    
    } 
    
    return good_matches;
}

vector<vector<DMatch>> PanoramicImage::cleaning(vector<Mat>& descriptors, vector<vector<KeyPoint>>&keypoints, vector<vector<DMatch>>& matches, bool doslopeFilter)
{
	double ratio = 3;
	vector<vector<DMatch>> good_matches;
    vector<vector<int>> key;
    vector<vector<int>> lookup;
	for (int i = 0; i < nrImg - 1; i++)
	{
		double min_dist = INFINITY;
        vector<double> distances;
        cv::Scalar mean_d,stddev_d;
		// calculation of max and min distances between keypoints in the i-th image
		for (int k = 0; k < descriptors[i].rows; k++)
		{
			double dist = matches[i][k].distance;
            distances.push_back(matches[i][k].distance);
			if (dist < min_dist)
				min_dist = dist;
		}
        cv::meanStdDev(distances, mean_d, stddev_d);
		vector<DMatch> gm;
        vector<DMatch> candidates;
        lookup.push_back(vector<int>());
		for (int k = 0; k < descriptors[i].rows; k++)
		{
            int filter = (matches[i][k].distance>(mean_d[0]-3*stddev_d[0])&&matches[i][k].distance <(mean_d[0]+3*stddev_d[0]));
            int position = keypoints[i][matches[i][k].queryIdx].pt.x;
			if (matches[i][k].distance < ratio * min_dist && position > 200)
            {
                lookup[i].push_back(1);
            }else{
                lookup[i].push_back(0);
            }
		}
	}
    //delegate the actual filter to the method
    good_matches = filters(descriptors,keypoints, matches, lookup);
	return good_matches;
}


void PanoramicImage::drawMatch(vector<Mat>& imagesOnCylinder, vector<vector<KeyPoint>>& keypoints, vector<vector<DMatch>>& good_matches)
{
	vector<Mat> img_matches(nrImg);
	for (int i = 0; i < nrImg - 1; i++)
	{
		drawMatches(imagesOnCylinder[i], keypoints[i], imagesOnCylinder[i + 1], keypoints[i + 1], good_matches[i], img_matches[i], Scalar::all(-1),
					Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		String name_img_matches = "Good Match " + to_string(i);
		imwrite("good_matches/" + name_img_matches + ".jpg", img_matches[i]);
	}
}

vector<Mat> PanoramicImage::getMaskFromHomography(vector<vector<DMatch>>& matches, vector<vector<KeyPoint>>& keypoints)
{
    vector<Mat> m;
    for (size_t i = 0; i < nrImg -1; i++)
    {
        //-- Localize the object
        std::vector<Point2f> obj;
        std::vector<Point2f> scene;
        for( int j = 0; j < matches[i].size(); j++ )
        {
            //-- Get the keypoints from the good matches
            obj.push_back( keypoints[i][matches[i][j].queryIdx].pt );
            scene.push_back( keypoints[i+1][matches[i][j].trainIdx].pt );
        }
        Mat mask;
        Mat H = findHomography( obj, scene, RANSAC, 1, mask); //openCV 4 uses RANSAC instead of CV_RANSAC
        m.push_back(mask);
    }
    return m;
}

void PanoramicImage::extractInlier(vector<vector<DMatch>>& matches, vector<Mat>& m,vector<vector<DMatch>>& inlier_matches)
{
    for (size_t i = 0; i < nrImg -1; i++)
    {
        vector<DMatch> im;
		for (int k = 0; k < matches[i].size(); k++)
        {
			if (m[i].at<uchar>(k))//m.type() is 0 -> CV_8U -> uchar
            {
                im.push_back(matches[i][k]);
            }

        }
		inlier_matches.push_back(im);
    }
}

//this method was taken from stackoverflow, it's immediate to see how it works at it was used because 
//it let me have more flexibility than the simple addWeighted() openCV method
void PanoramicImage::alphaBlend(Mat& img1, Mat&img2, Mat& mask, Mat& blended)
{
    // Blend img1 and img2 (of CV_8UC3) with mask (CV_8UC1)
    assert(img1.size() == img2.size() && img1.size() == mask.size());
    blended = cv::Mat(img1.size(), img1.type());
    for (int y = 0; y < blended.rows; ++y){
        for (int x = 0; x < blended.cols; ++x){
            float alpha = mask.at<unsigned char>(y, x)/255.0f;
            blended.at<cv::Vec3b>(y,x) = alpha*img1.at<cv::Vec3b>(y,x) + (1-alpha)*img2.at<cv::Vec3b>(y,x);
        }
    }
}

Mat PanoramicImage::stacking(vector<Mat>& imagesOnCylinder, vector<vector<DMatch>>& inlier_matches, vector<vector<KeyPoint>>& keypoints)
{
    double column_dim = 0;
	int row_dim = imagesOnCylinder[0].rows;
    //add robustness to the translation (take the mean of the shift's vectors)
    vector<double> avgDeltaX;
    vector<double> avgDeltaY;
    vector<int> h_pad_from_baseline;
    h_pad_from_baseline.push_back(0);
    for (size_t i = 0; i < nrImg - 1; i++)
    {
        vector<int> deltaX;
        vector<int> deltaY;
        double sumX = 0, sumY = 0;
        for (int k = 0; k < inlier_matches[i].size(); k++)
        {
            double par1 = keypoints[i][inlier_matches[i][k].queryIdx].pt.x - keypoints[i + 1][inlier_matches[i][k].trainIdx].pt.x;
            double par2 = keypoints[i][inlier_matches[i][k].queryIdx].pt.y - keypoints[i + 1][inlier_matches[i][k].trainIdx].pt.y;
            deltaX.push_back(par1);
            deltaY.push_back(par2);
            sumX = sumX + deltaX.back();
            sumY = sumY + deltaY.back();
        }
            
        avgDeltaX.push_back(sumX / deltaX.size());
        avgDeltaY.push_back(sumY / deltaY.size());
        h_pad_from_baseline.push_back(h_pad_from_baseline.back()+round(avgDeltaY.back()));
        column_dim = column_dim + avgDeltaX.back();
    }
    
    vector<int> l_pad_final;
    vector<int> h_pad_final;

    int maxElh_pad = *max_element(h_pad_from_baseline.begin(),h_pad_from_baseline.end());
    int minElh_pad = *min_element(h_pad_from_baseline.begin(),h_pad_from_baseline.end());

    for (size_t i = 0; i < h_pad_from_baseline.size(); i++)
    {
        l_pad_final.push_back(maxElh_pad-h_pad_from_baseline[i]);
        h_pad_final.push_back(minElh_pad-h_pad_from_baseline[i]);
    }
    vector<Mat> paddedImages;
    int h_slice = *min_element(h_pad_final.begin(),h_pad_final.end());
    int l_slice = *max_element(l_pad_final.begin(),l_pad_final.end());
 
    for (size_t i = 0; i < nrImg; i++)
    {
        Mat borderImage;
        copyMakeBorder(imagesOnCylinder[i], borderImage, abs(h_pad_final[i]),abs(l_pad_final[i]),0,0,BORDER_CONSTANT,Scalar(0));
        paddedImages.push_back(borderImage);
    } 
    column_dim = column_dim + imagesOnCylinder[nrImg - 1].cols;
    Mat result(paddedImages[0].rows, column_dim,imagesOnCylinder[0].type(),Scalar(0));
    paddedImages[0].copyTo(result.rowRange(0, paddedImages[0].rows).colRange(0, paddedImages[0].cols));
    double accumulation = 0;
    for (int i = 0; i < nrImg -1; i++)
    {
        accumulation = accumulation + avgDeltaX[i];
        Mat commonArea_left_image = paddedImages[i](Rect(avgDeltaX[i],0,imagesOnCylinder[0].cols-avgDeltaX[i],paddedImages[i].rows));
        Mat commonArea_right_image = paddedImages[i+1](Rect(0,0,imagesOnCylinder[0].cols-avgDeltaX[i], paddedImages[i+1].rows));
        //create the gradient for the blending of the common area
        Mat Mask(commonArea_right_image.size(), CV_8U, Scalar(0));
        for (int r = 0; r < Mask.cols; r++)
        {
            Mask.col(r).setTo(r);
        }
        Mat blended;
        alphaBlend(commonArea_right_image, commonArea_left_image,Mask,blended);
        blended.copyTo(paddedImages[i+1](Rect(0,0,imagesOnCylinder[0].cols-avgDeltaX[i], paddedImages[i+1].rows)));
        paddedImages[i+1].copyTo(result.rowRange(0, paddedImages[i+1].rows).colRange(accumulation, accumulation + paddedImages[i +1].cols));
    }
    Mat result_cropped = result.rowRange(abs(h_slice),row_dim).colRange(0,result.cols);
    imwrite("output_1_cropped_withMask.jpg",result_cropped);
    return result_cropped;
}

vector<Mat> PanoramicImage::equalizeLAB(vector<Mat> images)
{
    vector<Mat> equalizedImages;
    for (int i = 0; i < nrImg; i++)
    {
        Mat lab, equalizedL, equalizedImage, equalizedImageLAB;
        vector<Mat> componentsMatrix;
        cv::cvtColor(images[i], lab, cv::COLOR_BGR2Lab,3);
        split(lab,componentsMatrix);
        cv::equalizeHist(componentsMatrix[0],equalizedL);
        vector<Mat> hist_equalized = {equalizedL,componentsMatrix[1],componentsMatrix[2]};
        merge(hist_equalized, equalizedImageLAB);
        cv::cvtColor(equalizedImageLAB, equalizedImage, cv::COLOR_Lab2BGR);
        equalizedImages.push_back(equalizedImage);
    }
    return equalizedImages;
}
Mat PanoramicImage::createPanoramicImage(bool advanced)
{
    vector<Mat> imagesOnCylinder = ToCylinder(angle,dirname, false, false);
    vector<vector<KeyPoint>> keypoints(nrImg);
    vector<Mat> descriptors(nrImg);
    if(advanced)
    {
        Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
        detectAndComputeKD(imagesOnCylinder, detector,keypoints,descriptors);
    }else
    {
        Ptr<ORB> detector = ORB::create();
        detectAndComputeKD(imagesOnCylinder, detector,keypoints,descriptors);
    }

    vector<vector<DMatch>> matches;
    hardmatch(matches, descriptors, advanced);
    vector<vector<DMatch>>good_matches = cleaning(descriptors, keypoints, matches, true);
    drawMatch(imagesOnCylinder, keypoints, good_matches);
    vector<Mat> mask = getMaskFromHomography(good_matches, keypoints);
    vector<vector<DMatch>> inlier_matches;
    extractInlier(good_matches,mask, inlier_matches);
    int robust = 1;
    imagesOnCylinder = ToCylinder(angle,dirname, true, true);
    Mat result;
    if(robust)
    {
        result = stacking(imagesOnCylinder, inlier_matches, keypoints);
        imwrite("outputWithMask.jpg",result);
    }
   return result;
}

