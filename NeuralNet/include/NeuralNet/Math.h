#pragma once

#include <NeuralNet/Tensor.h>
#include <NeuralNet/Padding.h>
#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

using namespace std;

namespace NN
{
	class Math
	{
	public:
        static void MatrixMultiplication(const Tensor& A, const Tensor& B, Tensor& output);

        static void Im2Col(const Tensor& input, Tensor& output,
            std::pair<size_t, size_t> kernelSize,
            std::pair<size_t, size_t> strides,
            Padding padding);

        static void Col2Im(const Tensor& columnGradient, Tensor& output,
            const std::vector<size_t>& inputShape,
            std::pair<size_t, size_t> kernelSize,
            std::pair<size_t, size_t> strides,
            Padding padding);


	protected:

	};
}