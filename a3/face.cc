#include "face.h"
#include <numeric>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#define CROPX 128
#define CROPY 140
#define OFFX 61
#define OFFY 45

Mat cropFace(const Mat3b& img)
{
    Mat _img, face;
    img.copyTo(_img);
    face = _img(Rect(OFFX, OFFY, CROPX, CROPY));
    return face;
}

float vsum(vector<float> vec)
{
    float sum = accumulate(vec.begin(), vec.end(), 0.0);
    return sum;
}

Mat preprocess(const Mat& img)
{
    Mat face = cropFace(img);
    cvtColor(face, face, COLOR_BGR2GRAY);
    equalizeHist(face, face);
    return face;
}

struct FACE::FACEPimpl
{
    SVM svm;
    HOGDescriptor hog;
    Mat1f descriptors, responses;
};

/// Constructor
FACE::FACE() : pimpl(new FACEPimpl())
{
    pimpl->hog.winSize = Size(CROPX, CROPY);
    pimpl->hog.blockSize = Size(int(CROPX / 2), int(CROPY / 2));
    pimpl->hog.blockStride = Size(int(CROPX / 4), int(CROPY / 4));
    pimpl->hog.cellSize = Size(int(CROPX / 4), int(CROPY / 4));
}

/// Destructor
FACE::~FACE()
{
}

/// Start the training.  This resets/initializes the model.
void FACE::startTraining()
{
}

/// Add a new person.
///
/// @param img1:  250x250 pixel image containing a scaled and aligned face
/// @param img2:  250x250 pixel image containing a scaled and aligned face
/// @param same: true if img1 and img2 belong to the same person
void FACE::train(const Mat3b& img1, const Mat3b& img2, bool same)
{
    Mat face1 = preprocess(img1);
    Mat face2 = preprocess(img2);

    vector<float> feat1, feat2;
    pimpl->hog.compute(face1, feat1);
    pimpl->hog.compute(face2, feat2);

    vector<float> diff;
    for (unsigned i = 0; i < feat1.size(); ++i)
    {
        diff.push_back(feat1[i] - feat2[i]);
    }

    Mat1f descriptor(1, diff.size(), &diff[0]);
    pimpl->descriptors.push_back(descriptor);
    pimpl->responses.push_back(Mat1f(1, 1, float(same)));
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
void FACE::finishTraining()
{
    SVMParams params;
    /* params.kernel_type = CvSVM::RBF; */
    /* params.gamma = 1.0; */
    /* params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 600, 0.01); */
    pimpl->svm.train(pimpl->descriptors, pimpl->responses, Mat(), Mat(), params);
}

/// Verify if img corresponds to the provided name.  The result is a floating point
/// value directly proportional to the probability of being correct.
///
/// @param img1:  250x250 pixel image containing a scaled and aligned face
/// @param img2:  250x250 pixel image containing a scaled and aligned face
/// @return:    similarity score between both images
double FACE::verify(const Mat3b& img1, const Mat3b& img2)
{
    Mat face1 = preprocess(img1);
    Mat face2 = preprocess(img2);
    Mat face2f = preprocess(img2);
    flip(face2f, face2f, 1);

    vector<float> feat1, feat2, feat2f;
    pimpl->hog.compute(face1, feat1);
    pimpl->hog.compute(face2, feat2);
    pimpl->hog.compute(face2f, feat2f);

    vector<float> diff1, diff2;
    for (unsigned i = 0; i < feat1.size(); ++i)
    {
        diff1.push_back(feat1[i] - feat2[i]);
        diff2.push_back(feat1[i] - feat2f[i]);
    }

    Mat1f descriptor1(1, diff1.size(), &diff1[0]);
    Mat1f descriptor2(1, diff2.size(), &diff2[0]);

    double out1 = -pimpl->svm.predict(descriptor1, true) + 100;
    double out2 = -pimpl->svm.predict(descriptor2, true) + 100;

    return (out1 > out2 ? out1 : out2);
}

