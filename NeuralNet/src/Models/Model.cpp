#include <NeuralNet/Models/Model.h>

namespace NN::Models
{
    Model::Model(std::vector<size_t> inputShape)
        : inputShape_(inputShape),
        optimizer_(OptimizerType::SGD, 0.001f)
    {
    }

    Model::Model(std::vector<size_t> inputShape, unsigned int seed)
        : inputShape_(inputShape),
        randomGenerator_(seed),
        optimizer_(OptimizerType::SGD, 0.001f)
    {
    }

    Model::Model(std::vector<size_t> inputShape, std::mt19937& generator)
        : inputShape_(inputShape), 
        randomGenerator_(generator),
        optimizer_(OptimizerType::SGD, 0.001f)
    {
    }
}