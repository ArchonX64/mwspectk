#include <fstream>
#include <vector>
#include <iterator>
#include <iostream>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <sstream>

#define NOW std::time(nullptr)

using std::cout;

int main(int argc, char** argv) {

    // Arguments
    std::string freq_path = argv[1];
    std::string inten_path = argv[2];
    int peak_num = std::stoi(argv[3]);
    float freq_var = std::stof(argv[4]);
    float max_doub_sep = std::stof(argv[5]);
    float max_cdp_sep = std::stof(argv[6]);
    float inten_rat_var = std::stof(argv[7]);

    // File setup
    std::ofstream log("log.txt");
    std::cout.rdbuf(log.rdbuf());
    std::ifstream freq_file(freq_path);
    std::ifstream inten_file(inten_path);

    // Array setup
    auto* peaks = new float[peak_num];
    auto* inten = new float[peak_num];

    // Frequnency array
    cout << "Reading file...     ";
    if (freq_file.is_open()) {
        for (int i = 0; i < peak_num; i++) {
            freq_file >> peaks[i];
        }
    } else {
        cout << "Couldn't read file!";
        return -1;
    }
    freq_file.close();

    // Intensity array
    if (inten_file.is_open()) {
        for (int i = 0; i < peak_num; i++) {
            inten_file >> inten[i];
        }
    } else {
        cout << "Couldn't read file!";
        return -1;
    }
    inten_file.close();
    cout << "Done" << std::endl;

    // Peak difference calculation
    std::time_t start = NOW;
    cout << "Beginning peak difference calculations..." << std::endl;
    float** storage = new float*[peak_num];
    for (int i = 0; i < peak_num; i++) {
        float* storage2 = new float[peak_num];
        std::fill(storage2 + 0, storage2 + peak_num,
            -1.0);
        storage[i] = storage2;

        for (int j = i + 1; j < peak_num; j++) {
            float diff = peaks[j] - peaks[i];

            if (NOW - start > 1) {
                cout << "Comparing peak " << i << " with peak " << j << std::endl;
                start = NOW;
            }

            if (diff > max_doub_sep) break;

            storage2[j] = diff;
        }
    }
    cout << "Finished" << std::endl;

    // CDP Finding
    cout << "Starting difference comparisons...";
    std::vector<int> cdps;
    for (int left1 = 0; left1 < peak_num - 3; left1++) {
        for (int left2 = left1 + 1; left2 < peak_num - 2; left2++) {
            for (int right1 = left2 + 1; right1 < peak_num - 1; right1++) {
                for (int right2 = right1 + 1; right2 < peak_num; right2++) {
                    if (NOW - start > 1) {
                        cout << "Comparing pair [" << left1 << ", " << left2 << "] with [" << right1 << ", "
                             << right2 << "], total peaks found is " << cdps.size() << std::endl;
                        start = NOW;
                    }

                    // Null values will be -1, do not consider them
                    if (storage[left1][left2] < 0 || storage[right1][right2] < 0) {
                        break;
                    }

                    // CDPS will exist if the spacing between doublets is the same within freq_var
                    if (std::abs(storage[right1][right2] - storage[left1][left2]) < freq_var) {
                        if (false) {
                            cout << storage[left1][left2] << " " << storage[right1][right2] << ", "
                            << " " << left1
                            << " " << left2
                            << " " << right1
                            << " " << right2
                            << ", " << peaks[left1]
                            << " " << peaks[left2]
                            << " " << peaks[right1]
                            << " " << peaks[right2] << std::endl;
                        }

                        // The ratio of intensities between doublets must also be similar
                        if (std::abs(inten[left1] / inten[right2] - 1) < inten_rat_var
                         && std::abs(inten[right1] / inten[left2] - 1) < inten_rat_var) {
                            cdps.push_back(left1);
                            cdps.push_back(left2);
                            cdps.push_back(right1);
                            cdps.push_back(right2);
                        }
                    }
                }
                // The doublets must be within a certain range of each other
                if (peaks[right1] - peaks[left2] > max_cdp_sep) {
                   break;
                }
            }
        }
    }
    cout << "Finished" << std::endl;

    cout << "Writing output file...";
    std::ofstream cdps_out("./cdps");
    std::ostream_iterator<int> output_iterator(cdps_out, "\n");
    std::copy(std::begin(cdps), std::end(cdps), output_iterator);
    cdps_out.close();
    log.close();
}