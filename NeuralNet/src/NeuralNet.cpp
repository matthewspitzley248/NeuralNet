// NeuralNet.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#pragma once

#include <iostream>
#include <NeuralNet/NeuralNet.h>
#include <filesystem>




namespace NN
{
    NeuralNet::NeuralNet(unsigned int seed)
        : seed_(seed),
        randomGenerator_(seed)
    {
    }

    void NeuralNet::Sequential()
    {
        model_ = std::make_unique<Models::Sequential>();
    }
}



//int main()
//{
//    cout << "Hello World!\n";
//
//    ImageLoader imgLoader = ImageLoader();
//
//    std::cout << std::filesystem::current_path() << '\n';
//
//    Tensor image = imgLoader.Load(R"(Datasets\Zero_full (1).jpg)");
//
//    NeuralNet::NeuralNet model = NeuralNet();
//}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
