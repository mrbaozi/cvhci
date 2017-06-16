#include "skinmodel.h"
#include <iostream>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

class SkinModel::SkinModelPimpl
{
    public:
        SkinModelPimpl()
        {
        };
        ~SkinModelPimpl();

        void train(const Mat3b& img, const Mat1b& mask)
        {
        }

        Mat preprocess(const Mat3b& img)
        {
            Mat _img;
            img.copyTo(_img);
            vector<Mat> channels;

            GaussianBlur(_img, _img, Size(3, 3), 0, 0);

            cvtColor(_img, _img, COLOR_BGR2HSV);

            split(_img, channels);
            equalizeHist(channels[2], channels[2]);
            merge(channels, _img);

            return _img;
        }

        Mat removeContours(const Mat1b& skin)
        {
            Mat _skin;
            skin.copyTo(_skin);

             // find contours
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(_skin, contours, hierarchy,
                    CV_RETR_EXTERNAL,
                    CV_CHAIN_APPROX_NONE,
                    Point(0, 0));

            // remove small and large contours
            vector<int> idx;
            for (unsigned i = 0; i < contours.size(); ++i)
            {
                if (contours[i].size() < 40 || contours[i].size() > 800)
                {
                    contours.erase(contours.begin() + i);
                    --i;
                }
            }

            Mat mask = Mat::zeros(skin.size(), CV_8UC3);
            drawContours(mask, contours, -1, Scalar(255, 255, 255), -1);

            cvtColor(mask, mask, CV_BGR2GRAY);

            return mask;
       }

        Mat postprocess(const Mat1b& skin)
        {
            Mat _skin;
            skin.copyTo(_skin);

            erode(_skin, _skin, Mat(), Point(-1, -1), 1);

            Mat mask = removeContours(_skin);

            dilate(mask, mask, Mat(), Point(-1, -1), 3);

            // connect areas
            int morph_size = 5;
            int morph_elem = 0;
            Mat element = getStructuringElement(
                    morph_elem,
                    Size(2*morph_size+1, 2*morph_size+1),
                    Point(morph_size, morph_size));
            morphologyEx(mask, mask, MORPH_CLOSE, element, Point(-1, -1), 1);

            GaussianBlur(mask, mask, Size(17, 17), 0, 0);

            return mask;
        }

        Mat1b classify(const Mat3b& img)
        {
            Mat hsv = preprocess(img);

            vector<Mat> channels;
            split(hsv, channels);

            threshold(channels[0], channels[0], 23, UCHAR_MAX, CV_THRESH_BINARY_INV);
            threshold(channels[1], channels[1], 44, UCHAR_MAX, CV_THRESH_BINARY);
            threshold(channels[2], channels[2], 120, UCHAR_MAX, CV_THRESH_BINARY);

            erode(channels[0], channels[0], Mat(), Point(-1, -1), 1);

            Mat1b skin(img.size(), CV_8UC1);
            bitwise_and(channels[0], channels[1], skin);
            bitwise_and(skin, channels[2], skin);

            Mat skin_out = postprocess(skin);

            /* imshow("img", img); */
            /* imshow("skin", skin); */
            /* imshow("skin_out", skin_out); */
            /* waitKey(0); */

            return skin_out;
        };
};

/// Constructor
SkinModel::SkinModel()
{
    pimpl = new SkinModelPimpl();
}

/// Destructor
SkinModel::~SkinModel()
{
}

/// Start the training.  This resets/initializes the model.
///
/// Implementation hint:
/// Use this function to initialize/clear data structures used for training the skin model.
void SkinModel::startTraining()
{
    //--- IMPLEMENT THIS ---//
}

/// Add a new training image/mask pair.  The mask should
/// denote the pixels in the training image that are of skin color.
///
/// @param img:  input image
/// @param mask: mask which specifies, which pixels are skin/non-skin
void SkinModel::train(const Mat3b& img, const Mat1b& mask)
{
    //--- IMPLEMENT THIS ---//
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
///
/// Implementation hint:
/// e.g normalize w.r.t. the number of training images etc.
void SkinModel::finishTraining()
{
    //--- IMPLEMENT THIS ---//
}


/// Classify an unknown test image.  The result is a probability
/// mask denoting for each pixel how likely it is of skin color.
///
/// @param img: unknown test image
/// @return:    probability mask of skin color likelihood
Mat1b SkinModel::classify(const Mat3b& img)
{
    //--- IMPLEMENT THIS ---//
    return pimpl->classify(img);
}
