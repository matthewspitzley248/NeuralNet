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

	unique_ptr<LayerContext> Dropout::CreateContext() const
	{
		if (!randomGenerator_)
			throw runtime_error("Dropout RNG has not been initialized.");

		auto context = make_unique<DropoutContext>();
		context->randomGenerator.seed((*randomGenerator_)());

		return context;
	}

	void Dropout::Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape)
	{
		randomGenerator_ = &generator;
	}

	std::vector<size_t> Dropout::GetOutputShape(const std::vector<size_t>& inputShape)
	{
		return inputShape;
	}

	void Dropout::RunBatch(ImageBatch& batch, const size_t batchSize, std::vector<std::unique_ptr<LayerContext>>& contexts, WorkerPool& workerPool)
	{
		vector<WorkerPool::Job> jobs;
		jobs.reserve(batchSize);

		imageContexts_.clear();
		imageContexts_.resize(batchSize);
		//imageContexts_.reserve(batchSize);

		//For each image in the batch add it to the jobs list for the threads or workers to handle
		for (size_t i = 0; i < batchSize; i++)
		{
			jobs.emplace_back([this, &batch, &contexts, i](size_t workerIndex)
				{
					ImageLabel& image = batch.imageLabels[i];

					//Create a new Context for this image and add it to imageContexts
					imageContexts_[i] = DropoutImageContext();

					//Forward pass using this worker's contexts and image context
					Run(image.image, *contexts[workerIndex], imageContexts_[i]);
				});
		}

		workerPool.SubmitJobs(std::move(jobs));
		workerPool.WaitForJobs();
	}

	void Dropout::Run(Tensor& source, LayerContext& gradientContext, DropoutImageContext& imageContext)
	{
		auto& context = dynamic_cast<DropoutContext&>(gradientContext);

		const size_t size = source.GetSize();

		//Resize mask to match the input tensor.
		imageContext.dropoutMask.resize(size);

		//No dropout during inference.
		if (!training_)
		{
			return;
		}

		uniform_real_distribution<float> distribution(0.0f, 1.0f);

		float* data = source.GetData().data();

		for (size_t i = 0; i < size; i++)
		{
			const bool keep = distribution(context.randomGenerator) >= dropoutRate_;

			imageContext.dropoutMask[i] = keep;

			if (keep)
			{
				data[i] *= scale_;
			}
			else
			{
				data[i] = 0.0f;
			}
		}
	}

	void Dropout::Run(Tensor& source, LayerContext& context)
	{
	}

	void Dropout::BackpropagateBatch(vector<Tensor>& batchGradients, const size_t batchSize, std::vector<std::unique_ptr<LayerContext>>& contexts, WorkerPool& workerPool)
	{
		vector<WorkerPool::Job> jobs;
		jobs.reserve(batchSize);

		//For each image in the batch add it to the jobs list for the threads or workers to handle
		for (size_t i = 0; i < batchSize; i++)
		{
			jobs.emplace_back([this, &batchGradients, &contexts, i](size_t workerIndex)
				{
					//pass using this worker's contexts.
					Backpropagate(batchGradients[i], *contexts[workerIndex], imageContexts_[i]);
				});
		}

		workerPool.SubmitJobs(std::move(jobs));
		workerPool.WaitForJobs();
	}

	void Dropout::Backpropagate(Tensor& gradient, LayerContext& gradientContext, DropoutImageContext& imageContext)
	{
		const size_t size = gradient.GetSize();

		if (imageContext.dropoutMask.size() != size)
		{
			throw runtime_error("Dropout mask size does not match gradient size.");
		}

		float* gradientData = gradient.GetData().data();

		for (size_t i = 0; i < size; i++)
		{
			if (imageContext.dropoutMask[i])
			{
				gradientData[i] *= scale_;
			}
			else
			{
				gradientData[i] = 0.0f;
			}
		}
	}

	void Dropout::Backpropagate(Tensor& gradient, LayerContext& context)
	{
	}

	void Dropout::ApplyGradients(Optimizer& optimizer, int batchSize)
	{
	}

	void Dropout::ResetGradients(LayerContext& context)
	{
	}

	void Dropout::AccumulateGradients(LayerContext& context)
	{
	}
}