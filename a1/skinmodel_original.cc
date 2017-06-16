#include "skinmodel.h"
#include <cmath>
#include <iostream>

using namespace std;
using namespace cv;

/// Constructor
SkinModel::SkinModel()
{
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
    Mat1b skin = Mat1b::zeros(img.rows, img.cols);

    //--- IMPLEMENT THIS ---//
    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {

            if (false)
                skin(row, col) = rand()%256;

            if (false)
                skin(row, col) = img(row,col)[2];

            if (true) {

                Vec3b bgr = img(row, col);
                if (bgr[2] > bgr[1] && bgr[1] > bgr[0])
                    skin(row, col) = 2*(bgr[2] - bgr[0]);
            }
        }
    }

    return skin;
}

