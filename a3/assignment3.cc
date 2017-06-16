///
///  Assignment 3
///  Face Verification
///
///  Group Number:
///  Authors:
///
#define _OPENCV_FLANN_HPP_
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <fstream>

#include "face.h"
#include "ROC.h"

using namespace std;

int main(int argc, char* argv[]) {

    /// parse command line options
    boost::program_options::variables_map pom;
    {
        namespace po = boost::program_options;
        po::options_description pod(string("Allowed options for ")+argv[0]);
        pod.add_options()
            ("help,h", "produce this help message")
            ("gui,g", "Enable the GUI")
            ("out,o", po::value<string>(), "Path where to write the results")
            ("path", po::value<string>()->default_value("data"), "Path where to read input data");

        po::positional_options_description pop;
        pop.add("path", -1);

        po::store(po::command_line_parser( argc, argv ).options(pod).positional(pop).run(), pom);
        po::notify(pom);

        if (pom.count("help")) {
            cout << "Usage:" << endl <<  pod << "\n";
            return 0;
        }
    }

    /// get image filenames
    string path = pom["path"].as<string>();
    vector<string> people;
    {
        namespace fs = boost::filesystem;
        for (fs::directory_iterator it(fs::path(path+"/")); it!=fs::directory_iterator(); it++) {
            if (is_regular_file(*it) and it->path().filename().stem().string().back()=='1') {
                string filename = it->path().filename().string();

                people.push_back(filename.substr(0,filename.size()-5));
            }
        }
    }
    //	srand(time(0));
    random_shuffle(people.begin(), people.end());

    /// Perform a 5 fold cross evaluation, store results on a ROC
    ROC<int> roc;
    constexpr uint nFolds = 5;
    for (uint fold=0; fold<nFolds; fold++) {

        random_shuffle(people.begin(), people.end());

        vector<pair<string,string>> peoplepairs;
        for (uint i = 0; i < people.size()/2; i++)
            peoplepairs.push_back({people[i], people[(i+1)%(people.size()/2)]});
        for (uint i = people.size()/2; i < people.size(); i++)
            peoplepairs.push_back({people[i], people[i]});

        random_shuffle(peoplepairs.begin(), peoplepairs.end());

        vector<pair<string,string>> train, test;
        for (uint i=0; i<peoplepairs.size(); i++) {
            if (i%nFolds==fold) {
                test.push_back(peoplepairs[i]);
            } else {
                train.push_back(peoplepairs[i]);
            }
        }

        /// create person classification model instance
        FACE model;

        /// train model with all images in the train folder
        cout << "Start Training" << endl;
        model.startTraining();

        for (auto &p : train) {
            cout << "Train on person " << p.first << "-" << p.second << endl;
            model.train( cv::imread(path+"/"+p.first+"1.jpg",-1), cv::imread(path+"/"+p.second+"2.jpg",-1), p.first == p.second );
        }

        cout << "Finish Training" << endl;
        model.finishTraining();

        /// test model with all images in the test folder,
        for (auto &p : test) {
            double hyp = model.verify( cv::imread(path+"/"+p.first+"1.jpg",-1), cv::imread(path+"/"+p.second+"2.jpg",-1));
            roc.add(p.first == p.second, hyp);
            cout << "Test: " << p.first << "-" << p.second << ": " << hyp << endl;
        }
    }

    /// After testing, update statistics and show results
    roc.update();

    cout << "Best EER score: " << 1 - roc.EER << endl;

    /// Display final result if desired
    if (pom.count("gui")) {
        cv::imshow("ROC", roc.draw());
        cv::waitKey(0);
    }

    /// Ouput a summary of the data if required
    if (pom.count("out")) {

        string p = pom["out"].as<string>();

        /// GRAPH format with one FPR and TPR coordinates per line
        ofstream graph(p+"/graph.txt");
        for (auto &dot : roc.graph)
            graph << dot.first << " " << dot.second << endl;

        /// Single output of the 1-EER score
        ofstream score(p+"/score.txt");
        score << (1-roc.EER) << endl;
        /// Ouput of the obtained ROC figure
        cv::imwrite(p+"/ROC.png", roc.draw());
    }
}

