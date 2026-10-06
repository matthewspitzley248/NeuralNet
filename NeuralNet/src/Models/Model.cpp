#include <NeuralNet/Models/Model.h>

namespace NN::Models
{
    Model::Model(unsigned int seed)
        : randomGenerator_(seed)
    {
    }


    Model::Model(std::mt19937& generator)
        : randomGenerator_(generator)
    {
    }
}