#include <NeuralNet/Loss.h>

namespace NN
{

	float Loss::Calculate(const Tensor& prediction, int targetLabel, LossType lossType) const
	{
		switch (lossType)
		{
		case LossType::CategoricalCrossEntropy:
			return CategoricalCrossEntropy(prediction, targetLabel);
		}

		throw std::runtime_error("Unsupported loss function");
	}


	Tensor Loss::CalculateGradient(const Tensor& prediction, int targetLabel, LossType lossType) const
	{
		switch (lossType)
		{
		case LossType::CategoricalCrossEntropy:
			return CategoricalCrossEntropyGradient(prediction, targetLabel);
		}

		throw std::runtime_error("Unsupported loss function");
	}


	float Loss::CategoricalCrossEntropy(const Tensor& prediction, int targetLabel) const
	{
		const size_t predSize = prediction.GetSize();
		if (targetLabel < 0 || targetLabel >= static_cast<int>(predSize))
		{
			throw std::out_of_range("Invalid target label");
		}

		constexpr float epsilon = 1e-10f;

		// Extract raw data pointer to bypass internal lookup overhead
		const float* predData = prediction.GetData().data();
		float probability = predData[targetLabel];

		// Prevent log(0).
		probability = std::clamp(probability, epsilon, 1.0f);

		return -std::log(probability);
	}

	Tensor Loss::CategoricalCrossEntropyGradient(const Tensor& prediction, int targetLabel) const
	{
		const size_t predSize = prediction.GetSize();
		if (targetLabel < 0 || targetLabel >= static_cast<int>(predSize))
		{
			throw std::out_of_range("Invalid target label");
		}

		constexpr float epsilon = 1e-10f;

		Tensor gradient(prediction.GetShape());

		const float* predData = prediction.GetData().data();
		float* gradData = gradient.GetData().data();

		float probability = std::clamp(predData[targetLabel], epsilon, 1.0f);

		// Optimization: Use standard library bulk memory zeroing 
		// instead of an explicit for-loop element reset.
		std::fill_n(gradData, predSize, 0.0f);

		// Directly write to the target label index using raw array access
		gradData[targetLabel] = -1.0f / probability;

		return gradient;
	}
}