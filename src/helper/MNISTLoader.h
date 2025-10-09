#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

class MNISTLoader {
private:
  vector<vector<double>> images;
  vector<vector<double>> labels;

public:
  void load(const string &filepath) {
    images.clear();
    labels.clear();

    fstream file(filepath);
    string line;
    while (getline(file, line)) {
      vector<string> values = _split(line, ',');
      double target = stod(values[0]);
      vector<double> label(10, 0.0);
      label[(int)target] = 1.0;
      labels.push_back(label);

      vector<double> image(28 * 28);
      for (int i = 0; i < 28 * 28; i++) {
        image[i] = stod(values[i + 1]) > 0 ? 1 : 0;
      }
      images.push_back(image);
    }

    cout << "------------------------" << endl;
    cout << "Total images: " << images.size() << endl;
    cout << "Total labels: " << labels.size() << endl;
    cout << "Data Loaded." << endl;
    cout << "------------------------" << endl;

    _print();
  };

  vector<vector<double>> getImages() { return images; }
  vector<vector<double>> getLabels() { return labels; }

  void printData(vector<double> &image, vector<double> &label) {
    cout << "label: ";
    for (auto l : label) {
      cout << l << " ";
    }
    cout << endl;

    string chars = ".#";

    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        int char_idx = image[i * 28 + j] * (chars.size() - 1);
        cout << chars[char_idx] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

private:
  void _print() {
    cout << "Print first 3 images:" << endl;
    for (int i = 0; i < 3; i++) {
      cout << "image [" << i << "]" << endl;
      printData(images[i], labels[i]);
    }
    cout << "------------------------" << endl;
  }

  std::vector<std::string> _split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(s);
    while (std::getline(iss, token, delimiter)) {
      tokens.push_back(token);
    }
    return tokens;
  }
};
