#include "hog.h"

using namespace std;
using namespace cv;

#define WINX 63
#define WINY 126

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

    pimpl->hog.winSize = Size(63, 126);
    pimpl->hog.blockSize = Size(18, 18);
    pimpl->hog.blockStride = Size(9, 9);
    pimpl->hog.cellSize = Size(6, 6);
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
    Mat3b img1 = img(Rect((img.cols-WINX)/2, (img.rows-WINY)/2, WINX, WINY));
    Mat3b _img;
    cvtColor(img1, _img, COLOR_BGR2HSV);
    /* vector<Mat> ch; */
    /* split(_img, ch); */
    /* equalizeHist(ch[2], ch[2]); */
    /* merge(ch, _img); */
    cvtColor(_img, _img, COLOR_HSV2BGR);
    return _img;
}

/// Add a new training image.
///
/// @param img:  input image
/// @param bool: value which specifies if img represents a person
void HOG::train(const Mat3b& img, bool isPerson)
{
    Mat3b img2 = preprocess(img);

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
    params.kernel_type = CvSVM::RBF;
    params.gamma = 1.0;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 600, 0.01);
    pimpl->svm.train(pimpl->descriptors, pimpl->responses, Mat(), Mat(), params);
}

/// Classify an unknown test image.  The result is a floating point
/// value directly proportional to the probability of being a person.
///
/// @param img: unknown test image
/// @return:    probability of human likelihood
double HOG::classify(const Mat3b& img)
{
    Mat3b img2 = preprocess(img);

    vector<float> vDescriptor;
    pimpl->hog.compute(img2, vDescriptor);
    Mat1f descriptor(1, vDescriptor.size(), &vDescriptor[0]);

    return -pimpl->svm.predict(descriptor, true);
}

