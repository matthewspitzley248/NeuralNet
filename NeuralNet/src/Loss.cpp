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
		if (targetLabel < 0 || targetLabel >= static_cast<int>(prediction.GetSize()))
		{
			throw std::out_of_range("Invalid target label");
		}

		constexpr float epsilon = 1e-10f;

		float probability = prediction.At(targetLabel);

		// Prevent log(0).
		probability = std::clamp(probability, epsilon, 1.0f);

		return -std::log(probability);
	}

	Tensor Loss::CategoricalCrossEntropyGradient(const Tensor& prediction, int targetLabel) const
	{
		if (targetLabel < 0 || targetLabel >= static_cast<int>(prediction.GetSize()))
		{
			throw std::out_of_range("Invalid target label");
		}

		constexpr float epsilon = 1e-10f;

		Tensor gradient(prediction.GetShape());

		float probability = std::clamp(prediction.At(targetLabel), epsilon, 1.0f);

		for (int i = 0; i < static_cast<int>(prediction.GetSize()); ++i)
		{
			gradient.At(i) = 0.0f;
		}

		gradient.At(targetLabel) = -1.0f / probability;

		return gradient;
	}
}