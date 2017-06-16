#include "hog.h"

using namespace std;
using namespace cv;

class HOG::HOGPimpl {
    public:
        Mat1f descriptors;
        Mat1f responses;

        SVM svm;
        HOGDescriptor hog;
};


/// Constructor
HOG::HOG()
{
    pimpl = std::shared_ptr<HOGPimpl>(new HOGPimpl());
}

/// Destructor
HOG::~HOG()
{
}

/// Start the training.  This resets/initializes the model.
void HOG::startTraining()
{
}

/// Add a new training image.
///
/// @param img:  input image
/// @param bool: value which specifies if img represents a person
void HOG::train(const Mat3b& img, bool isPerson)
{
    Mat3b img2 = img(Rect((img.cols-64)/2, (img.rows-128)/2, 64, 128));
    vector<float> vDescriptor;
    pimpl->hog.compute(img2, vDescriptor);
    Mat1f descriptor(1, vDescriptor.size(), &vDescriptor[0]);

    pimpl->descriptors.push_back(descriptor);
    pimpl->responses.push_back(Mat1f(1, 1, float(isPerson)));
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
void HOG::finishTraining()
{
    SVMParams params;
    pimpl->svm.train(pimpl->descriptors, pimpl->responses, Mat(), Mat(), params);
}

/// Classify an unknown test image.  The result is a floating point
/// value directly proportional to the probability of being a person.
///
/// @param img: unknown test image
/// @return:    probability of human likelihood
double HOG::classify(const Mat3b& img)
{
    Mat3b img2 = img(Rect((img.cols-64)/2, (img.rows-128)/2, 64, 128));

    vector<float> vDescriptor;
    pimpl->hog.compute(img2, vDescriptor);
    Mat1f descriptor(1, vDescriptor.size(), &vDescriptor[0]);

    return -pimpl->svm.predict(descriptor, true);
}

