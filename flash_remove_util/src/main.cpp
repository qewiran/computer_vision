#include "../hdr/flashrm.hpp"

int main()
{
    using namespace cv;
    std::string filename;
    std::cin >> filename;
    std::vector<size_t> flashesIdx = getBadFramesIdxes(filename);
    std::cout << flashesIdx.size() << '\n';
}