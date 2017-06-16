#include "skinmodel.h"
#include <iostream>
#include <numeric>
#include <assert.h>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

class SkinModel::SkinModelPimpl
{
    public:
        SkinModelPimpl()
        {
            int sizes[] = {256, 256, 256};
            map = new Mat(3, sizes, CV_16UC2);
            total_skin = 0;
            total_non_skin = 0;
        };
        ~SkinModelPimpl();

        int add_training_skin_pixel(Vec3b bgr, bool is_skin) {
            Vec<unsigned short, 2>* color_point = &(map->at <Vec<unsigned short, 2> > (bgr[0], bgr[1], bgr[2]));
            if (is_skin) {
                total_skin++;
                (*color_point)[0] += 1;
            } else {
                total_non_skin++;
            }
            (*color_point)[1] += 1;

            return total_skin + total_non_skin;
        }

        double add_training_non_skin_pixel() {
            return total_non_skin++;
        }

        /// Returns the probability for Skin (S) for a given color (X), so P(S|X).
        double test_skin_pixel(Vec3b bgr)
        {
            double P_S_X, P_X_S, P_X, P_S; // P(S|X), P(X|S), P(X), P(S)
            assert(total_skin > 0);
            assert(total_non_skin > 0);
            Vec<unsigned short, 2> color_point = map->at<Vec<unsigned short, 2> >(bgr[0], bgr[1], bgr[2]);

            P_X_S = (double)color_point[0] / (double)total_skin;
            P_X = (double)color_point[1] / (double)(total_non_skin + total_skin);
            P_S = (double)total_skin / (double)(total_non_skin + total_skin);
            P_S_X = (P_X != 0) ? (P_X_S/P_X)*P_S : 0;

            return P_S_X; // = P(S|X)
        }

        void train(const Mat3b& img, const Mat1b& mask)
        {
            train_hsv(img, mask);
            train_bcf(img, mask);
        }

        void equalizeRGB(Mat rgb)
        {
            vector<Mat> channels;
            split(rgb, channels);
            equalizeHist(channels[0], channels[0]);
            equalizeHist(channels[1], channels[1]);
            equalizeHist(channels[2], channels[2]);
            merge(channels, rgb);
        }

        Mat1b getSkin(vector<Mat> channels, int h, int s, int v, Size size)
        {
            threshold(channels[0], channels[0], h, UCHAR_MAX, CV_THRESH_BINARY_INV);
            threshold(channels[1], channels[1], s, UCHAR_MAX, CV_THRESH_BINARY);
            threshold(channels[2], channels[2], v, UCHAR_MAX, CV_THRESH_BINARY);

            erode(channels[0], channels[0], Mat(), Point(-1, -1), 1);

            Mat1b skin(size, CV_8UC1);
            bitwise_and(channels[0], channels[1], skin);
            bitwise_and(skin, channels[2], skin);

            return skin;
        }

        void train_bcf(const Mat3b& img, const Mat1b& mask)
        {
            for (int row = 0; row < img.rows; ++row) {
                for (int col = 0; col < img.cols; ++col) {
                    add_training_skin_pixel(img(row,col), mask(row,col) == 255);
                }
            }
        }

        void train_hsv(const Mat3b& img, const Mat1b& mask)
        {
            Mat hsv;
            img.copyTo(hsv);

            equalizeRGB(hsv);

            cvtColor(hsv, hsv, COLOR_BGR2HSV);

            vector<Mat> channels;
            split(hsv, channels);
            equalizeHist(channels[2], channels[2]);
            merge(channels, hsv);

            Mat pskin;
            Mat1b _mask;
            mask.copyTo(_mask);
            erode(_mask, _mask, Mat(), Point(-1, -1), 1);
            bitwise_and(hsv, hsv, pskin, _mask);

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
                    2.2 * stdevs_temp[1],
                    1.5 * stdevs_temp[2]);
            upper = mean_temp + Scalar(
                    0.5 * stdevs_temp[0],
                    1.0 * stdevs_temp[1],
                    1.5 * stdevs_temp[2]);

            for (int i = 0; i < 3; ++i)
            {
                lower[i] = round(lower[i]);
                upper[i] = round(upper[i]);
                if (lower[i] < 0) lower[i] = 0;
                if (upper[i] > 255) upper[i] = 255;
            }
        }

        Mat removeContours(const Mat1b& skin, unsigned low, unsigned high)
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
                if (contours[i].size() < low || contours[i].size() > high)
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

        Mat1b classify(const Mat3b& img)
        {
            Mat1b skin_hsv = classify_hsv(img);
            Mat1b skin_thr = classify_thr(img);
            Mat1b skin_bcf = classify_bcf(img);

            imshow("hsv", skin_hsv);
            imshow("thr", skin_thr);
            imshow("bcf", skin_bcf);
            waitKey(0);

            // use skin_hsv if skin_thr is mostly empty
            Mat1b skin;
            if (mean(skin_thr)[0] < 1) skin = skin_hsv;
            else skin = skin_thr;

            // find large differences between skin_thr and skin_hsv
            Mat1b skin_bit;
            bitwise_xor(skin, skin_hsv, skin_bit);
            bitwise_and(skin, skin_bit, skin_bit);
            skin_bit = removeContours(skin_bit, 600, 1500);
            if (mean(skin_bit)[0] > 5) skin = skin_hsv;

            skin = removeContours(skin, 100, 1000);
            /* dilate(skin, skin, Mat(), Point(-1, -1), 2); */

            /* int morph_size = 10; */
            /* int morph_elem = 2; */
            /* Mat element = getStructuringElement( */
            /*         morph_elem, */
            /*         Size(2*morph_size+1, 2*morph_size+1), */
            /*         Point(morph_size, morph_size)); */
            /* morphologyEx(skin, skin, MORPH_CLOSE, element, Point(-1, -1), 1); */

            /* GaussianBlur(skin, skin, Size(25, 25), 0, 0); */

            return skin;
        }

        Mat1b classify_thr(const Mat3b& img)
        {
            Mat _img;
            img.copyTo(_img);
            vector<Mat> channels;
            GaussianBlur(_img, _img, Size(3, 3), 0, 0);
            cvtColor(_img, _img, COLOR_BGR2HSV);
            split(_img, channels);
            equalizeHist(channels[2], channels[2]);

            return getSkin(channels, 26, 39, 119, img.size());
        }

        Mat1b classify_bcf(const Mat3b& img)
        {
            Mat1b skin = Mat1b::zeros(img.rows, img.cols);
            Mat1d prob_field = Mat1d::zeros(img.rows, img.cols);

            // get the probability of skin per pixel
            for (int row = 0; row < img.rows; ++row) {
                for (int col = 0; col < img.cols; ++col) {
                    Vec3b bgr = img(row,col);
                    prob_field(row,col) = test_skin_pixel(bgr); //P(X|S)
                }
            }

            // Convert the probabilities to a value between 0 and 255
            for (int row = 0; row < img.rows; ++row) {
                for (int col = 0; col < img.cols; ++col) {
                    int scaled_val = (int)((prob_field(row, col)*255.0)+0.5);
                    if (scaled_val <= 255) {
                        skin(row,col) = scaled_val;
                    } else {
                        cout << "Warning: scaled_val is " << scaled_val << " (>255), image now has broken pixels." << endl;
                    }
                }
            }
        }

        Mat1b classify_hsv(const Mat3b& img)
        {
            Mat hsv;
            img.copyTo(hsv);

            GaussianBlur(hsv, hsv, Size(3, 3), 0, 0);

            equalizeRGB(hsv);

            cvtColor(hsv, hsv, COLOR_BGR2HSV);

            vector<Mat> channels;
            split(hsv, channels);
            equalizeHist(channels[2], channels[2]);
            merge(channels, hsv);

            Mat1b skin;
            inRange(hsv, lower, upper, skin);

            return skin;
        }

    private:
        vector<Scalar> means;
        vector<Scalar> stdevs;
        Scalar lower;
        Scalar upper;
        Mat* map;
        int total_skin;
        int total_non_skin;
};

/// Constructor
SkinModel::SkinModel()
{
    pimpl = new SkinModelPimpl();
}

/// Destructor
SkinModel::~SkinModel()
{
    delete pimpl;
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
