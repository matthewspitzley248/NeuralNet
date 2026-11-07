#include <NeuralNet/Optimizer.h>

namespace NN
{
	Optimizer::Optimizer(OptimizerType type, float learningRate)
		: type_(type),
		learningRate_(learningRate)
	{
	}

	void Optimizer::Apply(Tensor& parameters, const Tensor& gradients, int batchSize, AdamLayerContext& adamContext)
	{
		if (batchSize <= 0)
		{
			throw invalid_argument("Batch size must be greater than zero.");
		}

		if (parameters.GetShape() != gradients.GetShape())
		{
			throw invalid_argument("Parameters and gradients must have the same shape.");
		}

		const size_t totalSize = parameters.GetSize();
		float* paramData = parameters.GetData().data();
		const float* gradData = gradients.GetData().data();

		switch (type_)
		{
		case OptimizerType::SGD:
		{
			const float scale = 1.0f / static_cast<float>(batchSize);
			const float effectiveLearningRate = learningRate_ * scale;

			// Fully vectorized SGD step
			for (size_t i = 0; i < totalSize; ++i)
			{
				paramData[i] -= effectiveLearningRate * gradData[i];
			}
			break;
		}
		case OptimizerType::Adam:
		{
			Adam(parameters, gradients, batchSize, adamContext);
			break;
		}
		}
	}

	void Optimizer::Adam(Tensor& parameters, const Tensor& gradients, int batchSize, AdamLayerContext& adamContext)
	{
		const size_t paramSize = parameters.GetSize();
		float* paramData = parameters.GetData().data();
		const float* gradData = gradients.GetData().data();

		if (paramSize != gradients.GetSize()) {
			throw std::invalid_argument("Parameters and gradients must have identical sizes.");
		}

		if (adamContext.m.GetSize() != paramSize) {
			adamContext.m.ReshapeAndResize(parameters.GetShape());
			adamContext.v.ReshapeAndResize(parameters.GetShape());

			//Ensure underlying vectors are zeroed out completely
			std::fill(adamContext.m.GetData().begin(), adamContext.m.GetData().end(), 0.0f);
			std::fill(adamContext.v.GetData().begin(), adamContext.v.GetData().end(), 0.0f);
		}

		float* mData = adamContext.m.GetData().data();
		float* vData = adamContext.v.GetData().data();

		const float scale = 1.0f / static_cast<float>(batchSize);

		const float biasCorrection1 = 1.0f - std::pow(beta1_, t_);
		const float biasCorrection2 = 1.0f - std::pow(beta2_, t_);

		// Hoist learning rate bias scaling outside the core math execution loop
		const float stepSize = learningRate_ * (std::sqrt(biasCorrection2) / biasCorrection1);

		// High-Velocity Contiguous Adam Math Loop (Compiler SIMD Vectorization optimized)
		for (size_t i = 0; i < paramSize; ++i)
		{
			// Average the gradient across the batch footprint
			const float g = gradData[i] * scale;

			// Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
			mData[i] = beta1_ * mData[i] + (1.0f - beta1_) * g;

			// Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * g^2
			vData[i] = beta2_ * vData[i] + (1.0f - beta2_) * g * g;

			// Compute final parameter adjust update: param -= stepSize * m / (sqrt(v) + epsilon)
			paramData[i] -= stepSize * mData[i] / (std::sqrt(vData[i]) + epsilon_);
		}
	}

	void Optimizer::UpdateLearningRate(float learningRate)
	{
		learningRate_ = learningRate;
	}
}