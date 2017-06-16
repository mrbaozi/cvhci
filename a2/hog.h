#pragma once

#define _OPENCV_FLANN_HPP_
#include <opencv/cv.h>
#include <opencv2/ml/ml.hpp>
#include <memory>

class HOG {
    class HOGPimpl;
    std::shared_ptr<HOGPimpl> pimpl;
    public:
    /// Constructor
    HOG();

    /// Destructor
    ~HOG();

    /// Start the training.  This resets/initializes the model.
    void startTraining();

    /// Add a new training image.
    ///
    /// @param img:  input image
    /// @param bool: value which specifies if img represents a person
    void train(const cv::Mat3b& img, bool isPerson);

    /// Finish the training.  This finalizes the model.  Do not call
    /// train() afterwards anymore.
    void finishTraining();

    /// Classify an unknown test image.  The result is a floating point
    /// value directly proportional to the probability of being a person.
    ///
    /// @param img: unknown test image
    /// @return:    probability of human likelihood
    double classify(const cv::Mat3b& img);

    private:
};
