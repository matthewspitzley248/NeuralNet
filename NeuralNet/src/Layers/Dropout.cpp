#include <NeuralNet/Layers/Dropout.h>

namespace NN::Layers
{
	Dropout::Dropout(float dropoutRate)
		: dropoutRate_(dropoutRate)
	{
		if (dropoutRate < 0.0f || dropoutRate >= 1.0f)
			throw invalid_argument("Invalid dropout rate.");

		scale_ = 1 / (1 - dropoutRate);
	}

	unique_ptr<WorkerContext> Dropout::CreateWorkerContext() const
	{
		if (!randomGenerator_)
			throw runtime_error("Dropout RNG has not been initialized.");

		auto context = make_unique<DropoutWorkerContext>();
		context->randomGenerator.seed((*randomGenerator_)());

		return context;
	}

	unique_ptr<ImageContext> Dropout::CreateImageContext() const
	{
		return std::make_unique<DropoutImageContext>();
	}

	void Dropout::Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape)
	{
		randomGenerator_ = &generator;
	}

	std::vector<size_t> Dropout::GetOutputShape(const std::vector<size_t>& inputShape) const
	{
		return inputShape;
	}

	void Dropout::Run(Tensor& data, WorkerContext& workerContext, ImageContext& imageContext)
	{
		auto& dropoutWorkerContext = dynamic_cast<DropoutWorkerContext&>(workerContext);
		auto& dropoutImageContext = dynamic_cast<DropoutImageContext&>(imageContext);

		const size_t size = data.GetSize();

		//Resize mask to match the input tensor.
		dropoutImageContext.dropoutMask.resize(size);

		//No dropout during inference.
		if (!training_)
		{
			return;
		}

		uniform_real_distribution<float> distribution(0.0f, 1.0f);

		float* inData = data.GetData().data();

		for (size_t i = 0; i < size; i++)
		{
			const bool keep = distribution(dropoutWorkerContext.randomGenerator) >= dropoutRate_;

			dropoutImageContext.dropoutMask[i] = keep;

			if (keep)
			{
				inData[i] *= scale_;
			}
			else
			{
				inData[i] = 0.0f;
			}
		}
	}

	void Dropout::Backpropagate(Tensor& gradient, WorkerContext& workerContext, ImageContext& imageContext)
	{
		auto& dropoutImageContext = dynamic_cast<DropoutImageContext&>(imageContext);

		const size_t size = gradient.GetSize();

		if (dropoutImageContext.dropoutMask.size() != size)
		{
			throw runtime_error("Dropout mask size does not match gradient size.");
		}

		float* gradientData = gradient.GetData().data();

		for (size_t i = 0; i < size; i++)
		{
			if (dropoutImageContext.dropoutMask[i])
			{
				gradientData[i] *= scale_;
			}
			else
			{
				gradientData[i] = 0.0f;
			}
		}
	}
}