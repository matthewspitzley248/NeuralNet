#include <NeuralNet/Layers.h>

namespace NN::Layers
{
	unique_ptr<WorkerContext> Layer::CreateWorkerContext() const
	{
		return make_unique<WorkerContext>();
	}

	unique_ptr<ImageContext> Layer::CreateImageContext() const
	{
		return std::make_unique<ImageContext>();
	}

	void Layer::RunBatch(ImageBatch& batch, const size_t batchSize, std::vector<std::unique_ptr<WorkerContext>>& workerContexts, WorkerPool& workerPool)
	{
		//printf("[Layer %p] RunBatch START - Batch: %zu\n", this, batchSize);

		vector<WorkerPool::Job> jobs;
		jobs.reserve(batchSize);

		imageContexts_.clear();
		imageContexts_.resize(batchSize);
		//imageContexts_.reserve(batchSize);

		//For each image in the batch add it to the jobs list for the threads or workers to handle
		for (size_t i = 0; i < batchSize; i++)
		{
			jobs.emplace_back([this, &batch, &workerContexts, i](size_t workerIndex)
				{
					ImageLabel& image = batch.imageLabels[i];

					//Create a new Context for this image and add it to imageContexts
					imageContexts_[i] = CreateImageContext();

					//Forward pass using this worker's contexts and image context
					Run(image.image, *workerContexts[workerIndex], *imageContexts_[i]);
				});
		}

		workerPool.SubmitJobs(std::move(jobs));
		workerPool.WaitForJobs();

		//printf("[Layer %p] Forward complete - Batch: %zu, Contexts: %zu\n", this, batchSize, imageContexts_.size());
	}

	void Layer::Run(Tensor& data, WorkerContext& workerContext, ImageContext& imageContext)
	{
	}

	void Layer::Run(Tensor& data)
	{
	}

	void Layer::BackpropagateBatch(vector<Tensor>& batchGradients, const size_t batchSize, std::vector<std::unique_ptr<WorkerContext>>& workerContexts, WorkerPool& workerPool)
	{
		//printf("[Layer %p] Backprop - Batch: %zu, Contexts: %zu\n", this, batchSize, imageContexts_.size());

		vector<WorkerPool::Job> jobs;
		jobs.reserve(batchSize);

		//For each image in the batch add it to the jobs list for the threads or workers to handle
		for (size_t i = 0; i < batchSize; i++)
		{
			jobs.emplace_back([this, &batchGradients, &workerContexts, i](size_t workerIndex)
				{
					//pass using this worker's contexts.
					Backpropagate(batchGradients[i], *workerContexts[workerIndex], *imageContexts_[i]);
				});
		}

		workerPool.SubmitJobs(std::move(jobs));
		workerPool.WaitForJobs();
	}

	void Layer::ApplyGradients(Optimizer& optimizer, int batchSize)
	{
	}

	void Layer::ResetGradients(WorkerContext& workerContext)
	{
	}

	void Layer::AccumulateGradients(WorkerContext& workerContext)
	{
	}
}
