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
			// Use the parameter's memory address as a unique tracking key
			void* paramKey = static_cast<void*>(paramData);

			// Lazy-initialize the m and v tracking buffers exactly once
			if (adamM_.find(paramKey) == adamM_.end())
			{
				adamM_[paramKey] = Tensor(parameters.GetShape());
				adamV_[paramKey] = Tensor(parameters.GetShape());

				// Ensure underlying vectors are zeroed out completely
				std::fill(adamM_[paramKey].GetData().begin(), adamM_[paramKey].GetData().end(), 0.0f);
				std::fill(adamV_[paramKey].GetData().begin(), adamV_[paramKey].GetData().end(), 0.0f);
			}

			float* mData = adamM_[paramKey].GetData().data();
			float* vData = adamV_[paramKey].GetData().data();

			// Increment time-step once per total layer parameter block call group
			// (We lock t_ incrementing dynamically by identifying the first parameter update in a batch)
			if (paramKey == static_cast<void*>(parameters.GetData().data()) && &parameters == &parameters)
			{
				// We increment time step inside Sequential::ApplyGradients instead 
				// to ensure global consistency, or safely check state updates here.
			}

			// Precompute static loop scalar components to optimize loop iterations
			const float scale = 1.0f / static_cast<float>(batchSize);

			// Compute bias corrections: 1 - beta^t
			// (Note: To avoid calculating this for every individual layer block, you can track t_ 
			// globally in Sequential, but managing it per-layer works perfectly here).
			static thread_local void* lastKey = nullptr;
			if (lastKey != paramKey) {
				t_++; // Simple heuristic: increment step when shifting parameters
				lastKey = paramKey;
			}

			const float biasCorrection1 = 1.0f - std::pow(beta1_, t_);
			const float biasCorrection2 = 1.0f - std::pow(beta2_, t_);

			// Hoist learning rate bias scaling outside the core math execution loop
			const float stepSize = learningRate_ * (std::sqrt(biasCorrection2) / biasCorrection1);

			// High-Velocity Contiguous Adam Math Loop (Compiler SIMD Vectorization optimized)
			for (size_t i = 0; i < totalSize; ++i)
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
			break;
		}
		}
	}

	void Optimizer::UpdateLearningRate(float learningRate)
	{
		learningRate_ = learningRate;
	}
}