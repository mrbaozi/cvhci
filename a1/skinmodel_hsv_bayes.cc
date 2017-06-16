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
#define DILATE_SIZE 3

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

class SkinModel::SkinModelPimpl
{
    public:
        SkinModelPimpl()
        {
            bcf = new CvNormalBayesClassifier();
            update = false;
        };
        ~SkinModelPimpl();

        void train(const Mat3b& img, const Mat1b& mask)
        {
            train_hsv(img, mask);
            train_bcf(img, mask);
        }

        void train_bcf(const Mat3b& img, const Mat1b& mask)
        {
            Mat3b rgb;
            img.copyTo(rgb);

            GaussianBlur(rgb, rgb, Size(3, 3), 0, 0);

            vector<Mat> channels;
            split(rgb, channels);
            equalizeHist(channels[0], channels[0]);
            equalizeHist(channels[1], channels[1]);
            equalizeHist(channels[2], channels[2]);
            merge(channels, rgb);

            Mat trainingData(rgb.rows * rgb.cols, 3, CV_32FC1);
            Mat trainingClasses(mask.rows * mask.cols, 1, CV_32FC1);
            for (int i = 0; i < rgb.rows * rgb.cols; ++i)
            {
                trainingData.at<float>(i, 0) = float(rgb.at<Vec3b>(i)[0]);
                trainingData.at<float>(i, 1) = float(rgb.at<Vec3b>(i)[1]);
                trainingData.at<float>(i, 2) = float(rgb.at<Vec3b>(i)[2]);
                trainingClasses.at<float>(i) = float(mask.at<uchar>(i));
            }

            bcf->train(trainingData, trainingClasses, Mat(), Mat(), update);
            update = true;
        }

        void train_hsv(const Mat3b& img, const Mat1b& mask)
        {
            Mat hsv;
            img.copyTo(hsv);

            vector<Mat> channels;
            split(hsv, channels);
            equalizeHist(channels[0], channels[0]);
            equalizeHist(channels[1], channels[1]);
            equalizeHist(channels[2], channels[2]);
            merge(channels, hsv);

            cvtColor(hsv, hsv, COLOR_BGR2HSV);

            split(hsv, channels);
            equalizeHist(channels[2], channels[2]);
            merge(channels, hsv);

            Mat pskin;
            bitwise_and(hsv, hsv, pskin, mask);

            vector<double> skinh;
            vector<double> skins;
            vector<double> skinv;
            for (int i = 0; i < pskin.rows * pskin.cols; ++i)
            {
                if (pskin.at<Vec3b>(i) != Vec3b(0, 0, 0) &&
                        pskin.at<Vec3b>(i)[0] < 50)
                {
                    skinh.push_back(pskin.at<Vec3b>(i)[0]);
                    skins.push_back(pskin.at<Vec3b>(i)[1]);
                    skinv.push_back(pskin.at<Vec3b>(i)[2]);
                }
            }

            vector<vector<double>> skinhsv;
            skinhsv.push_back(skinh);
            skinhsv.push_back(skins);
            skinhsv.push_back(skinv);

            vector<double> mean_res;
            vector<double> stdev_res;
            for (auto const& val: skinhsv)
            {
                double sum = accumulate(val.begin(), val.end(), 0.0);
                double mean = sum / double(val.size());
                double sq = inner_product(val.begin(), val.end(), val.begin(), 0.0);
                double stdev = sqrt(sq / double(val.size()) - mean * mean);
                mean_res.push_back(round(mean));
                stdev_res.push_back(round(stdev));
            }

            means.push_back(Scalar(mean_res[0], mean_res[1], mean_res[2]));
            stdevs.push_back(Scalar(stdev_res[0], stdev_res[1], stdev_res[2]));

            Scalar mean_temp = Scalar(0., 0., 0.);
            for (auto const& val: means)
            {
                mean_temp += val;
            }
            mean_temp /= double(means.size());

            Scalar stdevs_temp = Scalar(0., 0., 0.);
            for (auto const& val: stdevs)
            {
                stdevs_temp += val;
            }
            stdevs_temp /= double(stdevs.size());

            lower = mean_temp - Scalar(
                    1.5 * stdevs_temp[0],
                    1.9 * stdevs_temp[1],
                    1.5 * stdevs_temp[2]);
            upper = mean_temp + Scalar(
                    1.4 * stdevs_temp[0],
                    2.9 * stdevs_temp[1],
                    1.5 * stdevs_temp[2]);

            for (int i = 0; i < 3; ++i)
            {
                lower[i] = round(lower[i]);
                upper[i] = round(upper[i]);
                if (lower[i] < 0) lower[i] = 0;
                if (upper[i] > 255) upper[i] = 255;
            }
        }

        Mat1b classify(const Mat3b& img)
        {
            Mat1b skin_bcf = classify_bcf(img);
            Mat1b skin_hsv = classify_hsv(img);
            Mat1b skin;
            bitwise_and(skin_bcf, skin_hsv, skin);

            // find contours
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            Mat skin_contours;
            skin.copyTo(skin_contours);
            findContours(skin_contours, contours, hierarchy,
                    CV_RETR_EXTERNAL,
                    CV_CHAIN_APPROX_NONE,
                    Point(0, 0));

            // find small contours
            vector<int> idx;
            for (unsigned i = 0; i < contours.size(); ++i)
            {
                if (contours[i].size() < 50)
                {
                    idx.push_back(i);
                }
            }

            // delete small contours
            int count = 0;
            for (auto const& id: idx)
            {
                contours.erase(contours.begin() + id - count);
                ++count;
            }

            Mat mask = Mat::zeros(skin.size(), CV_8UC3);
            drawContours(mask, contours, -1, Scalar(255, 255, 255), -1);

            int morph_size = 10;
            int morph_elem = 0;
            Mat element = getStructuringElement(
                    morph_elem,
                    Size(2*morph_size+1, 2*morph_size+1),
                    Point(morph_size, morph_size));
            morphologyEx(mask, mask, MORPH_CLOSE, element, Point(-1, -1), 1);

            GaussianBlur(mask, mask, Size(25, 25), 0, 0);

            cvtColor(mask, mask, CV_BGR2GRAY);

            return mask;
        }

        Mat1b classify_bcf(const Mat3b& img)
        {
            Mat3b rgb;
            img.copyTo(rgb);

            GaussianBlur(rgb, rgb, Size(3, 3), 0, 0);

            vector<Mat> channels;
            split(rgb, channels);
            equalizeHist(channels[0], channels[0]);
            equalizeHist(channels[1], channels[1]);
            equalizeHist(channels[2], channels[2]);
            merge(channels, rgb);

            Mat predicted = Mat(rgb.rows, rgb.cols, CV_32F);
            for (int i = 0; i < rgb.rows * rgb.cols; ++i)
            {
                Mat sample = Mat(1, 3, CV_32FC1);
                sample.at<float>(0) = rgb.at<Vec3b>(i)[0];
                sample.at<float>(1) = rgb.at<Vec3b>(i)[1];
                sample.at<float>(2) = rgb.at<Vec3b>(i)[2];
                predicted.at<float>(i) = bcf->predict(sample);
            }

            Mat1b skin;
            predicted.copyTo(skin);

            return skin;
        };

        Mat1b classify_hsv(const Mat3b& img)
        {
            Mat hsv;
            img.copyTo(hsv);

            GaussianBlur(hsv, hsv, Size(3, 3), 0, 0);

            vector<Mat> channels;
            split(hsv, channels);
            equalizeHist(channels[0], channels[0]);
            equalizeHist(channels[1], channels[1]);
            equalizeHist(channels[2], channels[2]);
            merge(channels, hsv);

            cvtColor(hsv, hsv, COLOR_BGR2HSV);

            split(hsv, channels);
            equalizeHist(channels[2], channels[2]);
            merge(channels, hsv);

            Mat1b skin;
            inRange(hsv, lower, upper, skin);

            return skin;
        }

    private:
        CvNormalBayesClassifier* bcf;
        bool update;
        vector<Scalar> means;
        vector<Scalar> stdevs;
        Scalar lower;
        Scalar upper;
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
    return pimpl->classify(img);
}
