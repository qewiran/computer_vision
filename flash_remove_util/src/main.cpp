#include "../hdr/flashrm.hpp"
#include <iostream>
#include <opencv2/core/utility.hpp>

int main(int argc, char** argv)
{
    using namespace cv;
    std::string filename = argv[1];
    std::vector<size_t> flashesIdx = getBadFramesIdxes(filename);
    std::cout << flashesIdx.size() << '\n';
    // std::cout<<getBuildInformation()<<"\n";
}
