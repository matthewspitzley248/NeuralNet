#include <NeuralNet/Optimizer.h>

namespace NN
{
	Optimizer::Optimizer(OptimizerType type, float learningRate)
		: type_(type),
		learningRate_(learningRate)
	{
	}

	void Optimizer::Apply(Tensor& parameters, const Tensor& gradients, int batchSize)
	{
		if (batchSize <= 0)
		{
			throw invalid_argument("Batch size must be greater than zero.");
		}

		if (parameters.GetShape() != gradients.GetShape())
		{
			throw invalid_argument("Parameters and gradients must have the same shape.");
		}

		switch (type_)
		{
		case OptimizerType::SGD:
		{
			const float scale = 1.0f / static_cast<float>(batchSize);

			for (size_t i = 0; i < parameters.GetSize(); ++i)
			{
				parameters.AtFlat(i) -=
					learningRate_ * gradients.AtFlat(i) * scale;
			}

			break;
		}
		case OptimizerType::Adam:
		{

		}
		}
	}

	void Optimizer::UpdateLearningRate(float learningRate)
	{
		learningRate_ = learningRate;
	}
}