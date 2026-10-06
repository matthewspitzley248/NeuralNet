#pragma once

#include <NeuralNet/Layers/Layer.h>

namespace NN::Layers
{
    class Dense : public Layer
    {
    public:
        /// <summary>
        /// Dense layer based on Keras
        /// </summary>
        /// <param name="units">Dimensionality of the output space.</param>
        /// <param name="activation">Activation function to use.</param>
        Dense(
            int units,
            Activation activation = Activation::None
        );

        Tensor Run(const Tensor& input) override;
    protected:
        int units_;
        Activation activation_;
    };
}