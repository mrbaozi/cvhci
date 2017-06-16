#include "skinmodel.h"
#include <cmath>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

using std::cout;
using std::endl;

class SkinModel::SkinModelPimpl
{
    public:
        SkinModelPimpl()
        {
        };
        ~SkinModelPimpl();

        void train(const Mat3b& img, const Mat1b& mask)
        {
        };

        void classify(const Mat3b& img, Mat1b& skin)
        {
        };

        Mat1b GetSkin(Mat const &src) {
            // allocate the result matrix
            Mat dst = src.clone();

            split(dst, channels);
            equalizeHist(channels[0], channels[0]);
            equalizeHist(channels[1], channels[1]);
            equalizeHist(channels[2], channels[2]);
            merge(channels, dst);


            Vec3b cwhite = Vec3b::all(255);
            Vec3b cblack = Vec3b::all(0);

            Mat src_ycrcb, src_hsv;
            // OpenCV scales the YCrCb components, so that they
            // cover the whole value range of [0,255], so there's
            // no need to scale the values:
            cvtColor(src, src_ycrcb, CV_BGR2YCrCb);
            // OpenCV scales the Hue Channel to [0,180] for
            // 8bit images, so make sure we are operating on
            // the full spectrum from [0,360] by using floating
            // point precision:
            src.convertTo(src_hsv, CV_32FC3);
            cvtColor(src_hsv, src_hsv, CV_BGR2HSV);
            // Now scale the values between [0,255]:
            normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

            for (int i = 0; i < src.rows; i++)
            {
                for (int j = 0; j < src.cols; j++)
                {
                    Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
                    int B = pix_bgr.val[0];
                    int G = pix_bgr.val[1];
                    int R = pix_bgr.val[2];
                    // apply rgb rule
                    bool a = R1(R,G,B);

                    Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
                    int Y = pix_ycrcb.val[0];
                    int Cr = pix_ycrcb.val[1];
                    int Cb = pix_ycrcb.val[2];
                    // apply ycrcb rule
                    bool b = R2(Y,Cr,Cb);

                    Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
                    float H = pix_hsv.val[0];
                    float S = pix_hsv.val[1];
                    float V = pix_hsv.val[2];
                    // apply hsv rule
                    bool c = R3(H,S,V);

                    if (!(a&&b&&c))
                    {
                        dst.ptr<Vec3b>(i)[j] = cblack;
                    }
                }
            }

            dst.convertTo(dst, CV_8UC1);
            return dst;
        }

    private:
        bool R1(int R, int G, int B) {
            bool e1 = (R>95) && (G>40) && (B>20) && ((max(R,max(G,B)) - min(R, min(G,B)))>15) && (abs(R-G)>15) && (R>G) && (R>B);
            bool e2 = (R>220) && (G>210) && (B>170) && (abs(R-G)<=15) && (R>B) && (G>B);
            return (e1||e2);
        }

        bool R2(float Y, float Cr, float Cb) {
            bool e3 = Cr <= 1.5862*Cb+20;
            bool e4 = Cr >= 0.3448*Cb+76.2069;
            bool e5 = Cr >= -4.5652*Cb+234.5652;
            bool e6 = Cr <= -1.15*Cb+301.75;
            bool e7 = Cr <= -2.2857*Cb+432.85;
            return e3 && e4 && e5 && e6 && e7;
        }

        bool R3(float H, float S, float V) {
            return (H<25) || (H > 230);
        }
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
    return pimpl->GetSkin(img);
}

