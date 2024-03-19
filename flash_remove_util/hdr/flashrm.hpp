#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <stdexcept>
#include <vector>
#include <tuple>
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <thread>

cv::Mat lumCalc(const cv::Mat&);

std::tuple<cv::Mat, cv::Mat> lumDiff(const cv::Mat&, const cv::Mat&);

double lumAvgDiff(const cv::Mat&, const cv::Mat&);

std::vector<size_t> getBadFramesIdxes(cv::String);