//Layers.h : This is a file to make includes easier as it grabs all the NeuralNet files.

#pragma once
#include <random>

#include <NeuralNet/Layers.h>
#include <NeuralNet/Models.h>
#include <NeuralNet/Tensor.h>
#include <NeuralNet/ImageLoader.h>
#include <NeuralNet/Models/Sequential.h>


namespace NN
{
    class NeuralNet
    {
    public:
        NeuralNet(unsigned int seed = 0);

        void Sequential();

        Tensor Run(const Tensor& input);

    private:
        unsigned int seed_;
        std::mt19937 randomGenerator_;

        std::unique_ptr<Models::Model> model_;
    };
}