#include "../hdr/flashrm.hpp"

class VideoReadingException : public std::exception {
public:
    std::string what() {
        return "frame was not read!\n";
    }
};


cv::Mat lumCalc(const cv::Mat& img)
{
    using namespace cv;
    Size s = img.size();
    Mat lum = Mat(s, CV_32FC1);
    for (size_t j = 0; j < s.height; j++)
    {
        for (size_t k = 0; k < s.width; k++)
        {
            lum.at<float>(j, k) = 16 + 65.738 * double(img.at<Vec3b>(j, k)[2]) / 256.0 +
                129.057 * double(img.at<Vec3b>(j, k)[1]) / 256.0 + 25.064 * double(img.at<Vec3b>(j, k)[0]) / 256.0;
        }
    }
    return lum;
}


std::tuple<cv::Mat, cv::Mat> lumDiff(const cv::Mat& lumCurr, const cv::Mat& lumPrev)
{
    using namespace cv;
    Size s = lumCurr.size();
    Mat lumDelta = Mat(s, CV_32FC1);
    Mat lumPos = Mat(s, CV_32FC1);
    Mat lumNeg = Mat(s, CV_32FC1);

    lumDelta = lumCurr - lumPrev;

    for (size_t j = 0; j < s.height; j++)
    {
        for (size_t k = 0; k < s.width; k++)
        {
            if (lumDelta.at<float>(j, k) > 1e-6)
            {
                lumPos.at<float>(j, k) = lumDelta.at<float>(j, k);
                lumNeg.at<float>(j, k) = 0;
            }
            else
            {
                lumNeg.at<float>(j, k) = -lumDelta.at<float>(j, k);
                lumPos.at<float>(j, k) = 0;
            }
        }
    }
    return std::make_tuple(lumNeg, lumPos);
}


double lumAvgDiff(const cv::Mat& lumCurr, const cv::Mat& lumPrev)
{
    using namespace cv;
    auto t = lumDiff(lumCurr, lumPrev);
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

    if (hPos.at<float>(0, histRequiredSize) >= 1e-6 && hNeg.at<float>(0, histRequiredSize) >= 1e-6)
        return std::max(cv::mean(hPos)[0], cv::mean(hNeg)[0]);
    else if (hPos.at<float>(0, histRequiredSize) >= 1e-6 && hNeg.at<float>(0, histRequiredSize) < 1e-6)
        return cv::mean(hPos)[0];
    else if (hNeg.at<float>(0, histRequiredSize) >= 1e-6 && hPos.at<float>(0, histRequiredSize) < 1e-6)
        return cv::mean(hNeg)[0];
    else return 0.0;
}


std::vector<size_t> getBadFramesIdxes(cv::String filename)
{
    using namespace cv;
    VideoCapture cap(filename);

    try
    {
        if (!cap.isOpened())
        {
            throw VideoReadingException();
        }
    }
    catch (VideoReadingException& ex)
    {
        std::cout << ex.what() << "\n";
    }

    uint totalFrames = cap.get(CAP_PROP_FRAME_COUNT);
    uint fps = cap.get(CAP_PROP_FPS);

    double lumMeanPrev;

    std::vector<size_t> flashesIndex;
    std::vector<Mat> badFrames;

    Mat framePrev, frameCurr, L_prev, L_curr;

    double t = (double)getTickCount();

    for (size_t j = 0; j < totalFrames; j++)
    {
        try
        {
            if (j == 0)
            {
                if (cap.read(framePrev))
                {
                    L_prev = lumCalc(framePrev);
                    auto t = lumDiff(L_prev, Mat(L_prev.size(), CV_32FC1));
                    lumMeanPrev = cv::mean(L_prev)[0];
                }
                else throw VideoReadingException();
            }
            else
            {
                if (cap.read(frameCurr))
                {
                    Mat L_curr = lumCalc(frameCurr);

                    double lumDiffCurr = lumAvgDiff(L_curr, L_prev);
                    double lumMeanCurr = cv::mean(L_curr)[0];

                    if (std::min(lumMeanCurr, lumMeanPrev) <= 160 && lumDiffCurr >= 20.0)
                    {
                            // writer << j << ": " << lumDiffCurr << " " << lumMeanPrev << " " << lumMeanCurr << '\n';
                            flashesIndex.push_back(j);
                            // badFrames.push_back(frameCurr);
                            // imshow("bad frames curr", frameCurr);
                            // imshow("bad frames prev", framePrev);
                            // waitKey(0);
                            // using namespace std::chrono_literals;
                            // std::this_thread::sleep_for((5000ms));
                    }

                    lumMeanPrev = lumMeanCurr;
                    L_prev = L_curr.clone();
                    framePrev = frameCurr.clone();
                }
                else throw VideoReadingException();
            }
            std::cout << "\r loading... : " << j + 1 << " out of " << totalFrames - 1
                << " frames is processed" << ", time passed: "
                << ((double)getTickCount() - t) / getTickFrequency() << " seconds" << std::flush;
        }
        catch (VideoReadingException& ex)
        {
            std::cout << ex.what() << '\n';
        }

    }
    t = ((double)getTickCount() - t) / getTickFrequency();
    std::cout << "\n Times passed in seconds: " << t << '\n';

    return flashesIndex;
}
