#include "skinmodel.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

#define ERODE_SIZE 1
#define DILATE_SIZE 4

void balance_white(cv::Mat mat) {
    double discard_ratio = 0.05;
    int hists[3][256];
    memset(hists, 0, 3*256*sizeof(int));

    for (int y = 0; y < mat.rows; ++y) {
        uchar* ptr = mat.ptr<uchar>(y);
        for (int x = 0; x < mat.cols; ++x) {
            for (int j = 0; j < 3; ++j) {
                hists[j][ptr[x * 3 + j]] += 1;
            }
        }
    }

    // cumulative hist
    int total = mat.cols*mat.rows;
    int vmin[3], vmax[3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 255; ++j) {
            hists[i][j + 1] += hists[i][j];
        }
        vmin[i] = 0;
        vmax[i] = 255;
        while (hists[i][vmin[i]] < discard_ratio * total)
            vmin[i] += 1;
        while (hists[i][vmax[i]] > (1 - discard_ratio) * total)
            vmax[i] -= 1;
        if (vmax[i] < 255 - 1)
            vmax[i] += 1;
    }

    for (int y = 0; y < mat.rows; ++y) {
        uchar* ptr = mat.ptr<uchar>(y);
        for (int x = 0; x < mat.cols; ++x) {
            for (int j = 0; j < 3; ++j) {
                int val = ptr[x * 3 + j];
                if (val < vmin[j])
                    val = vmin[j];
                if (val > vmax[j])
                    val = vmax[j];
                ptr[x * 3 + j] = static_cast<uchar>((val - vmin[j]) * 255.0 / (vmax[j] - vmin[j]));
            }
        }
    }
}

class SkinModel::SkinModelPimpl
{
    public:
        SkinModelPimpl()
        {
        };
        ~SkinModelPimpl();

        void train(const Mat3b& img, const Mat1b& mask)
        {
            img.copyTo(rgb);

            cvtColor(rgb, hsv, COLOR_BGR2HSV);

            /* split(hsv, channels); */
            /* /1* equalizeHist(channels[0], channels[0]); *1/ */
            /* /1* equalizeHist(channels[1], channels[1]); *1/ */
            /* equalizeHist(channels[2], channels[2]); */
            /* merge(channels, hsv); */

            Mat pskin;
            bitwise_and(hsv, hsv, pskin, mask);

            for (int i = 0; i < pskin.rows; ++i)
            {
                for (int j = 0; j < pskin.cols; ++j)
                {
                    if (pskin.at<Vec3b>(i, j) != Vec3b(0, 0, 0) &&
                            pskin.at<Vec3b>(i, j)[0] < 50)
                    {
                        skinh.push_back(pskin.at<Vec3b>(i, j)[0]);
                        skins.push_back(pskin.at<Vec3b>(i, j)[1]);
                        skinv.push_back(pskin.at<Vec3b>(i, j)[2]);
                    }
                }
            }

            skinhsv.push_back(skinh);
            skinhsv.push_back(skins);
            skinhsv.push_back(skinv);

            vector<double> mean_res;
            vector<double> stdev_res;
            for (auto const& val: skinhsv
                )
            {
                double sum = accumulate(val.begin(), val.end(), 0.0);
                double mean = sum / val.size();
                double sq_sum = inner_product(val.begin(), val.end(),
                        val.begin(), 0.0);
                double stdev = sqrt(sq_sum / val.size() - mean * mean);
                mean_res.push_back(round(mean));
                stdev_res.push_back(round(stdev));
            }

            colormodel = Scalar(mean_res[0], mean_res[1], mean_res[2]);
            lower = colormodel - Scalar(2*stdev_res[0], 2*stdev_res[1], 2*stdev_res[2]);
            upper = colormodel + Scalar(2*stdev_res[0], 2*stdev_res[1], 2*stdev_res[2]);

            for (int i = 0; i < 3; ++i)
            {
                if (lower[i] < 0) lower[i] = 0;
                if (upper[i] > 255) upper[i] = 255;
            }
        }

        void classify(const Mat3b& img, Mat1b& skin)
        {
            img.copyTo(rgb);

            split(rgb, channels);
            GaussianBlur(channels[1], channels[1], Size(3, 3), 0, 0);
            merge(channels, rgb);

            cvtColor(rgb, hsv, COLOR_BGR2HSV);

            split(hsv, channels);
            equalizeHist(channels[2], channels[2]);
            merge(channels, hsv);

            inRange(hsv, lower, upper, skin);
            density_regularisation(img, skin);
            GaussianBlur(skin, skin, Size(19, 19), 0, 0);
            Mat skintmp;
            skin.copyTo(skintmp);
            bilateralFilter(skin, skintmp, 6, 12, 3);
            skintmp.copyTo(skin);

            /* imshow("rgb", rgb); */
            /* imshow("skin", skin); */
            /* waitKey(0); */
        };

        void density_regularisation(const Mat3b& img, Mat1b& imgFilter)
        {
            Mat sum;
            sum = Mat::zeros(img.rows, img.cols, CV_8UC1);
            uchar op;
            int erode, dilate;
            for (int i = 0; i < img.rows; i += 4) //Cycle over horizontal clusters
            {
                for (int j = 0; j < img.cols; j += 4) //Cycle over vertical clusters
                {
                    for (int k = 0; k < 4; k++) //Cycle horizontally within cluster
                    {
                        for (int l = 0; l < 4; l++) //Cycle vertically within cluster
                        {
                            if (imgFilter.at<uchar>(i + k, j + l) != 0) sum.at<uchar>(i, j)++;
                        }
                    }
                    if (sum.at<uchar>(i, j) == 0 || i == 0 || j == 0 || i == (img.rows - 4) || j == (img.cols - 4)) op = 0;
                    else if (sum.at<uchar>(i, j) > 0 &&  sum.at<uchar>(i, j) < 16) op = 128;
                    else op = 255;
                    for (int k = 0; k < 4; k++) //Cycle horizontally within cluster
                    {
                        for (int l = 0; l < 4; l++) //Cycle vertically within cluster
                        {
                            imgFilter.at<uchar>(i + k, j + l) = op;
                        }
                    }
                }
            }
            for (int i = 4; i < (img.rows - 4); i += 4) //Cycle over horizontal clusters
            {
                for (int j = 4; j < (img.cols -4); j += 4) //Cycle over vertical clusters
                {
                    erode = 0;
                    if (imgFilter.at<uchar>(i, j) == 255)
                    {
                        for (int k = -4; k < 5; k += 4)
                        {
                            for (int l = -4; l < 5; l += 4)
                            {
                                if (imgFilter.at<uchar>(i + k, j + l) == 255) erode++;
                            }
                        }
                        if (erode < ERODE_SIZE)
                        {
                            for (int k = 0; k < 4; k++) //Cycle horizontally within cluster
                            {
                                for (int l = 0; l < 4; l++) //Cycle vertically within cluster
                                {
                                    imgFilter.at<uchar>(i + k, j + l) = 0;
                                }
                            }
                        }
                    }
                }
            }
            for (int i = 4; i < (img.rows - 4); i += 4) //Cycle over horizontal clusters
            {
                for (int j = 4; j < (img.cols - 4); j += 4) //Cycle over vertical clusters
                {
                    dilate = 0;
                    if (imgFilter.at<uchar>(i, j) < 255)
                    {
                        for (int k = -4; k < 5; k += 4)
                        {
                            for (int l = -4; l < 5; l += 4)
                            {
                                if (imgFilter.at<uchar>(i + k, j + l) == 255) dilate++;
                            }
                        }
                        if (dilate > DILATE_SIZE)
                        {
                            for (int k = 0; k < 4; k++) //Cycle horizontally within cluster
                            {
                                for (int l = 0; l < 4; l++) //Cycle vertically within cluster
                                {
                                    imgFilter.at<uchar>(i + k, j + l) = 255;
                                }
                            }
                        }
                    }
                    for (int k = 0; k < 4; k++) //Cycle horizontally within cluster
                    {
                        for (int l = 0; l < 4; l++) //Cycle vertically within cluster
                        {
                            if (imgFilter.at<uchar>(i + k, j + l) != 255) imgFilter.at<uchar>(i + k, j + l) = 0;
                        }
                    }
                }
            }
        }

    private:
        vector<Mat> channels;
        vector<int> skinh;
        vector<int> skins;
        vector<int> skinv;
        vector<vector<int>> skinhsv;
        Scalar colormodel;
        Scalar lower;
        Scalar upper;
        Mat hsv;
        Mat rgb;
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
    pimpl->train(img, mask);
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
    Mat1b skin;
    pimpl->classify(img, skin);
    return skin;
}
