#include "opencv2/core/mat.hpp"
#include <vector>
#include <tuple>
#include <unistd.h>

cv::Mat lumCalc(cv::Mat);

std::tuple<cv::Mat, cv::Mat> lumDiff(cv::Mat, cv::Mat);

double lumAvgDiff(cv::Mat, cv::Mat);

std::vector<size_t> getBadFramesIdxes(cv::String);
