#include "../hdr/flashrm.hpp"
#include <fstream>
#include <opencv2/highgui.hpp>
#include "opencv2/videoio.hpp"
#include <iostream>
#include <immintrin.h>
#include <opencv2/core/utility.hpp>
#include <tbb/tbb.h>
class VideoReadingException : public std::exception {
public:
  std::string what() { return "frame was not read!\n"; }
};
cv::Mat lumCalcParallel(cv::Mat img){

  using namespace cv;
  Size s = img.size();
  Mat lum = Mat(s,CV_16UC1);

  parallel_for_(Range(0,s.height*s.width),[&](const Range& range){
      for (int r= range.start; r<range.end; ++r){
        size_t i = r/s.width;
        size_t j = r % s.width;
        Vec3b valsImg = img.ptr<Vec3b>(i)[j];
        double val = 16. + valsImg[0] * 65.738 / 256. +
                     valsImg[1] * 129.057 / 256. + valsImg[2] * 25.064 / 256.;
        lum.ptr<ushort>(i)[j] = val;
      }
      });
  return lum;
}

std::tuple<cv::Mat, cv::Mat> lumDiffParallel(cv::Mat lumCurr, cv::Mat lumPrev) {
  using namespace cv;
  Size s = lumCurr.size();
  lumCurr.convertTo(lumCurr,CV_16SC1);
  lumPrev.convertTo(lumPrev,CV_16SC1);
  Mat lumDelta = Mat(s, CV_16SC1);
  Mat lumPos = Mat(s, CV_16SC1);
  Mat lumNeg = Mat(s, CV_16SC1);

  lumDelta = lumCurr - lumPrev;
  parallel_for_(Range(0, s.width * s.height), [&](const Range &range) {
    for (int r = range.start; r < range.end; ++r) {
      size_t i = r / s.width;
      size_t j = r % s.width;
      if (lumDelta.ptr<short>(i)[j] > 1e-6) {
        lumPos.ptr<short>(i)[j] = lumDelta.ptr<short>(i)[j];
        lumNeg.ptr<short>(i)[j] = 0;
      } else {
        lumNeg.ptr<short>(i)[j] = -lumDelta.ptr<short>(i)[j];
        lumPos.ptr<short>(i)[j] = 0;
      }
    }
  });
  return std::make_tuple(lumNeg, lumPos);
}
short lumAvgDiffParallel(cv::Mat lumCurr, cv::Mat lumPrev){

  using namespace cv;
  auto t = lumDiffParallel(lumCurr, lumPrev);
  Mat lumPos = std::get<1>(t), lumNeg = std::get<0>(t);

  Mat hPos, hNeg;
  Size s = lumPos.size();

  int hSize = s.height * s.width;

  hPos = lumPos.reshape(1, 1);
  hNeg = lumNeg.reshape(1, 1);

  sort(hPos, hPos, SORT_DESCENDING);
  sort(hNeg, hNeg, SORT_DESCENDING);

  size_t histRequiredSize = hSize / 4.;
  double nomNeg = 0., denNeg = 0., nomPos = 0., denPos = 0.;

  if (hPos.at<short>(0, histRequiredSize) >= 1e-6 &&
      hNeg.at<short>(0, histRequiredSize) >= 1e-6)
    return std::max(cv::mean(hPos)[0], cv::mean(hNeg)[0]);
  else if (hPos.at<short>(0, histRequiredSize) >= 1e-6 &&
           hNeg.at<short>(0, histRequiredSize) < 1e-6)
    return cv::mean(hPos)[0];
  else if (hNeg.at<short>(0, histRequiredSize) >= 1e-6 &&
           hPos.at<short>(0, histRequiredSize) < 1e-6)
    return cv::mean(hNeg)[0];
  else
    return 0.0;
}

std::vector<size_t> getBadFramesIdxes(cv::String filename) {
  using namespace cv;
  setNumThreads(8);
  VideoCapture cap(filename, cv::CAP_FFMPEG);

  size_t height=cap.get(CAP_PROP_FRAME_HEIGHT) ; 
  size_t width =cap.get(CAP_PROP_FRAME_WIDTH) ; 

  size_t frameSize = height*width;
  size_t totalFrames = cap.get(CAP_PROP_FRAME_COUNT);
  size_t fps = cap.get(CAP_PROP_FPS);

  // cv::Mat img(height,width, CV_8UC3);

  short lumMeanCurr, lumMeanPrev=0, lumDiff;
  std::vector<size_t> flashesIndex;
  double t = (double)getTickCount();

  std::ostringstream oss;
  oss << "ffmpeg -loglevel panic -i " << filename
      << " -c:v rawvideo -pix_fmt yuv420p out.yuv";

  std::system(oss.str().c_str());

  std::ifstream reader("out.yuv", std::ios::binary);
  Mat L(height, width, CV_8UC1);
  Mat L_prev;
  char luma[frameSize];

  for (size_t i = 0; i < totalFrames; ++i) {
    reader.read(luma, frameSize);
    reader.ignore(frameSize / 2);
    Mat L(height, width, CV_8UC1, luma);
    if (i > 0) {
      lumDiff = lumAvgDiffParallel(L, L_prev);
    }
    lumMeanCurr = cv::mean(L)[0];

    if (std::min(lumMeanCurr, lumMeanPrev) <= 160 && std::abs(lumDiff) >= 20.)
      flashesIndex.push_back(i);
    lumMeanPrev = lumMeanCurr;

    L_prev = L.clone();
    std::cout << "\r loading... : " << i << " out of " << totalFrames - 1
              << " frames is processed" << ", time passed: "
              << ((double)getTickCount() - t) / getTickFrequency() << " seconds"
              << std::flush;
  }

  t = ((double)getTickCount() - t) / getTickFrequency();
  std::cout << "\n Times passed in seconds: " << t << '\n';
  std::system("rm out.yuv");
  
  // cv::Mat frame;
  // namedWindow("TEST", WINDOW_GUI_NORMAL);
  // auto it = flashesIndex.cbegin();
  // auto end = flashesIndex.cend();
  // if (it!=end)
  // {
  //   for (size_t i =0; i< totalFrames; ++i){
  //     cap.read(frame);
  //     if (i==*it){
  //       imshow("TEST", frame);
  //       waitKey(300);
  //       ++it;
  //     }
  //   }
  // }
  return flashesIndex;
}
