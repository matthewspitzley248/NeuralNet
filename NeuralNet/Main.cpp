#pragma once

#include <iostream>
#include <NeuralNet/NeuralNet.h>
#include <filesystem>
using namespace NN::Models;

using namespace NN;
using namespace NN::Layers;
using namespace std;


int main()
{
    cout << "Hello World!\n";

    ImageLoader imgLoader = ImageLoader();

    cout << filesystem::current_path() << '\n';

    Tensor image = imgLoader.Load(R"(Datasets\Zero_full (1).jpg)");

    //NeuralNet model = NeuralNet();

    //model.Sequential();
    Sequential model = Sequential(10);

    model.Add<Flatten>();
    model.Add<Dense>(1000, Activation::Relu);
    model.Add<Dense>(100, Activation::Relu);
    //output layer
    model.Add<Dense>(10);
}