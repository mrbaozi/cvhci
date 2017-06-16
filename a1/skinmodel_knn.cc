#include "skinmodel.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

#define ERODE_SIZE 1
#define DILATE_SIZE 4

class SkinModel::SkinModelPimpl
{
    public:
        SkinModelPimpl()
        {
            K = 3;
        };
        ~SkinModelPimpl();

        void train(const Mat3b& img, const Mat1b& mask)
        {
            if (!trained)
            {
                Mat tD, tC;
                Mat trainingData(img.rows * img.cols, 3, CV_32FC1);
                Mat trainingClasses(mask.rows * mask.cols, 1, CV_32FC1);
                img.copyTo(tD);
                mask.copyTo(tC);
                tD.convertTo(tD, CV_32FC1);
                tC.convertTo(tC, CV_32FC1);
                tD.reshape(3, tD.rows * tD.cols);
                tC.reshape(1, tC.rows * tC.cols);

                for (int i = 0; i < tD.rows; ++i)
                {
                    trainingData.at<Vec3f>(i) = tD.at<Vec3f>(i);
                    trainingClasses.at<Vec3f>(i) = tC.at<Vec3f>(i);
                }

                knn = new CvKNearest(trainingData, trainingClasses, Mat(), false, K);
                trained = true;
            }
        }

        void classify(const Mat3b& img, Mat1b& skin)
        {
            Mat tD;
            Mat testData(img.rows * img.cols, 3, CV_32FC1);
            img.copyTo(tD);
            tD.convertTo(tD, CV_32FC1);
            tD.reshape(3, tD.rows * tD.cols);

            for (int i = 0; i < tD.rows; ++i)
            {
                testData.at<Vec3f>(i) = tD.at<Vec3f>(i);
            }

            Mat predicted(testData.rows, 1, CV_32FC1);
            for (int i = 0; i < testData.rows; i++)
            {
                const Mat sample = testData.row(i);
                predicted.at<Vec3f>(i) = knn->find_nearest(sample, K);
            }

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
        CvKNearest* knn;
        int K;
        vector<Mat> channels;
        vector<vector<int>> skinhsv;
        Mat hsv;
        Mat rgb;
        bool trained = false;
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
