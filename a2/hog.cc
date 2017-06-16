#include "hog.h"

using namespace std;
using namespace cv;

#define WINX 64
#define WINY 128

class HOG::HOGPimpl {
    public:
        Mat1f descriptors;
        Mat1f responses;

        SVM svm;
        CvANN_MLP mlp;
        HOGDescriptor hog;
};

/// Constructor
HOG::HOG()
{
    pimpl = shared_ptr<HOGPimpl>(new HOGPimpl());

    /* pimpl->hog.winSize = Size(64, 128); */
    /* pimpl->hog.blockSize = Size(32, 64); */
    /* pimpl->hog.blockStride = Size(16, 32); */
    /* pimpl->hog.cellSize = Size(8, 8); */

    pimpl->hog.winSize = Size(64, 128);
    pimpl->hog.blockSize = Size(16, 16);
    pimpl->hog.blockStride = Size(8, 8);
    pimpl->hog.cellSize = Size(8, 8);

    Mat layers = Mat(4, 1, CV_32SC1);
    layers.row(0) = Scalar(3780);
    layers.row(1) = Scalar(40);
    layers.row(2) = Scalar(2);
    layers.row(3) = Scalar(1);
    pimpl->mlp.create(layers);
}

/// Destructor
HOG::~HOG()
{
}

/// Start the training.  This resets/initializes the model.
void HOG::startTraining()
{
}

Mat3b preprocess(const Mat3b& img)
{
    Mat3b _img = img(Rect((img.cols-WINX)/2, (img.rows-WINY)/2, WINX, WINY));
    return _img;
}

/// Add a new training image.
///
/// @param img:  input image
/// @param bool: value which specifies if img represents a person
void HOG::train(const Mat3b& img, bool isPerson)
{
    Mat3b _img = preprocess(img);

    vector<float> vDescriptor;
    pimpl->hog.compute(_img, vDescriptor);
    Mat1f descriptor(1, vDescriptor.size(), &vDescriptor[0]);

    /* cout << vDescriptor.size(); exit(0); */

    pimpl->descriptors.push_back(descriptor);
    pimpl->responses.push_back(Mat1f(1, 1, float(isPerson)));
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
void HOG::finishTraining()
{
    /* SVMParams svmParams; */
    /* svmParams.kernel_type = CvSVM::RBF; */
    /* svmParams.gamma = 1.0; */
    /* svmParams.term_crit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 600, 0.01); */
    /* pimpl->svm.train(pimpl->descriptors, pimpl->responses, Mat(), Mat(), svmParams); */

    CvANN_MLP_TrainParams mlpParams;

    pimpl->mlp.train(pimpl->descriptors, pimpl->responses, Mat(), Mat(), mlpParams);
}

/// Classify an unknown test image.  The result is a floating point
/// value directly proportional to the probability of being a person.
///
/// @param img: unknown test image
/// @return:    probability of human likelihood
double HOG::classify(const Mat3b& img)
{
    Mat3b _img = preprocess(img);

    vector<float> vDescriptor;
    pimpl->hog.compute(_img, vDescriptor);
    Mat1f descriptor(1, vDescriptor.size(), &vDescriptor[0]);

    float predicted = 0;
    Mat1f response(1, 1, &predicted);

    pimpl->mlp.predict(descriptor, response);

    return predicted;
    /* return -pimpl->svm.predict(descriptor, true); */
    /* return predicted - pimpl->svm.predict(descriptor, true); */
}

