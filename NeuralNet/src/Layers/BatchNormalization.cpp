#include <NeuralNet/Layers/BatchNormalization.h>

namespace NN::Layers
{
	BatchNormalization::BatchNormalization(float momentum, float epsilon)
		: momentum_(momentum), epsilon_(epsilon)
	{
		if (momentum_ < 0.0f || momentum_ > 1.0f)
		{
			throw invalid_argument("BatchNormalization momentum must be between 0 and 1.");
		}

		if (epsilon_ <= 0.0f)
		{
			throw invalid_argument("BatchNormalization epsilon must be greater than zero.");
		}
	}

	unique_ptr<WorkerContext> BatchNormalization::CreateWorkerContext() const
	{
        auto context = make_unique<BatchNormalizationWorkerContext>();

        context->gammaGradients.ReshapeAndResize({ channels_ });
        context->betaGradients.ReshapeAndResize({ channels_ });

        return context;
	}

	unique_ptr<ImageContext> BatchNormalization::CreateImageContext() const
	{
		return std::make_unique<BatchNormalizationImageContext>();
	}

	void BatchNormalization::Initialize(std::mt19937& generator, const std::vector<size_t>& inputShape)
    {
        if (inputShape.empty())
            throw std::invalid_argument("BatchNormalization requires a non-empty input shape.");

        channels_ = inputShape.back();

        if (channels_ == 0)
            throw std::invalid_argument("BatchNormalization requires at least one channel.");

        //Resize the optimization lookup vectors to match total framework configuration footprints
        inferenceScale_.resize(channels_);
        inferenceOffset_.resize(channels_);

        //Initialize trainable parameters.
        gamma_.ReshapeAndResize({ channels_ });
        beta_.ReshapeAndResize({ channels_ });

        std::fill(gamma_.GetData().begin(), gamma_.GetData().end(), 1.0f);

        //Initialize moving statistics.
        movingMean_.ReshapeAndResize({ channels_ });
        movingVariance_.ReshapeAndResize({ channels_ });

        std::fill(movingVariance_.GetData().begin(), movingVariance_.GetData().end(), 1.0f);

        //Initialize batch statistics.
        batchMean_.ReshapeAndResize({ channels_ });
        batchVariance_.ReshapeAndResize({ channels_ });
        inverseStdDev_.ReshapeAndResize({ channels_ });

        //Initialize accumulated gradients.
        gammaGradients_.ReshapeAndResize({ channels_ });
        betaGradients_.ReshapeAndResize({ channels_ });

        batchBetaGradients_.ReshapeAndResize({ channels_ });
        batchGammaGradients_.ReshapeAndResize({ channels_ });
    }

	std::vector<size_t> BatchNormalization::GetOutputShape(const std::vector<size_t>& inputShape) const
	{
		return inputShape;
	}

	void BatchNormalization::RunBatch(ImageBatch& batch, const size_t batchSize, std::vector<std::unique_ptr<WorkerContext>>& workerContexts, WorkerPool& workerPool)
	{
        if (batchSize == 0)
            return;

        // Calculate the number of values contributing to each channel.
        const Tensor& firstImage = batch.imageLabels[0].image;

        const size_t valuesPerImage = firstImage.GetSize() / channels_;
        const size_t totalValues = valuesPerImage * batchSize;

        batchMean_.ReshapeAndResize({ channels_ });
        batchVariance_.ReshapeAndResize({ channels_ });
        inverseStdDev_.ReshapeAndResize({ channels_ });

        //Zero out accumulation blocks completely to prevent previous-batch memory leakage
        std::fill(batchMean_.GetData().begin(), batchMean_.GetData().end(), 0.0f);
        std::fill(batchVariance_.GetData().begin(), batchVariance_.GetData().end(), 0.0f);


        float* batchMean = batchMean_.GetData().data();
        float* batchVariance = batchVariance_.GetData().data();

        //Calculate the mean for each channel.
        for (size_t i = 0; i < batchSize; i++)
        {
            const Tensor& image = batch.imageLabels[i].image;
            const float* imageData = image.GetData().data();

            for (size_t j = 0; j < image.GetSize(); j++)
            {
                const size_t channel = j % channels_;
                batchMean[channel] += imageData[j];
            }
        }

        for (size_t c = 0; c < channels_; c++)
        {
            batchMean[c] /= static_cast<float>(totalValues);
        }

        // Calculate variance for each channel.
        for (size_t i = 0; i < batchSize; i++)
        {
            const Tensor& image = batch.imageLabels[i].image;
            const float* imageData = image.GetData().data();

            for (size_t j = 0; j < image.GetSize(); j++)
            {
                const size_t channel = j % channels_;
                const float difference = imageData[j] - batchMean[channel];

                batchVariance[channel] += difference * difference;
            }
        }

        float* inverseStdDev = inverseStdDev_.GetData().data();
        float* movingMean = movingMean_.GetData().data();
        float* movingVariance = movingVariance_.GetData().data();

        for (size_t c = 0; c < channels_; c++)
        {
            batchVariance[c] /= static_cast<float>(totalValues);

            inverseStdDev[c] = 1.0f / std::sqrt(batchVariance[c] + epsilon_);
        }

        // Update running statistics during training.
        if (training_)
        {
            for (size_t c = 0; c < channels_; c++)
            {
                movingMean[c] = momentum_ * movingMean[c] + (1.0f - momentum_) * batchMean[c];

                movingVariance[c] = momentum_ * movingVariance[c] + (1.0f - momentum_) * batchVariance[c];
            }
        }
        else
        {
            // Use running statistics during inference.
            for (size_t c = 0; c < channels_; c++)
            {
                batchMean[c] = movingMean[c];

                inverseStdDev[c] = 1.0f / std::sqrt(movingVariance[c] + epsilon_);
            }
        }

        //Reuse the base implementation for worker scheduling and contexts.
        Layer::RunBatch(batch, batchSize, workerContexts, workerPool);
	}

    void BatchNormalization::RunBatch(ImageBatch& batch, const size_t batchSize, WorkerPool& workerPool)
    {
        float* gamma = gamma_.GetData().data();
        float* beta = beta_.GetData().data();
        float* movingMean = movingMean_.GetData().data();
        float* movingVariance = movingVariance_.GetData().data();

        // Precompute inference parameters.
        for (size_t c = 0; c < channels_; c++)
        {
            const float scale = gamma[c] / std::sqrt(movingVariance[c] + epsilon_);

            inferenceScale_[c] = scale;

            inferenceOffset_[c] = beta[c] - movingMean[c] * scale;
        }

        // Distribute images to workers.
        Layer::RunBatch(batch, batchSize, workerPool);
    }

	void BatchNormalization::BackpropagateBatch(vector<Tensor>& batchGradients, const size_t batchSize, std::vector<std::unique_ptr<WorkerContext>>& workerContexts, WorkerPool& workerPool)
	{
        if (batchSize == 0)
            return;

        std::fill(batchBetaGradients_.GetData().begin(), batchBetaGradients_.GetData().end(), 0.0f);
        std::fill(batchGammaGradients_.GetData().begin(), batchGammaGradients_.GetData().end(), 0.0f);

        float* batchBetaGradients = batchBetaGradients_.GetData().data();
        float* batchGammaGradients = batchGammaGradients_.GetData().data();

        totalBatchValues_ = batchSize * (batchGradients[0].GetSize() / channels_);

        // Calculate the batch-wide gradient sums.
        for (size_t i = 0; i < batchSize; i++)
        {
            auto& context = dynamic_cast<BatchNormalizationImageContext&>(*imageContexts_[i]);

            const float* gradient = batchGradients[i].GetData().data();
            const float* normalized = context.normalized.GetData().data();

            for (size_t j = 0; j < batchGradients[i].GetSize(); j++)
            {
                const size_t channel = j % channels_;

                batchBetaGradients[channel] += gradient[j];

                batchGammaGradients[channel] += gradient[j] * normalized[j];
            }
        }

        //Reuse the base implementation for worker scheduling and contexts.
        Layer::BackpropagateBatch(batchGradients, batchSize, workerContexts, workerPool);
	}

	void BatchNormalization::Run(Tensor& data)
	{
        float* input = data.GetData().data();
        const size_t totalSize = data.GetSize();

        for (size_t i = 0; i < totalSize; i++)
        {
            const size_t channel = i % channels_;

            input[i] = input[i] * inferenceScale_[channel] + inferenceOffset_[channel];
        }
	}

	void BatchNormalization::Run(Tensor& data, WorkerContext& workerContext, ImageContext& imageContext)
	{
        auto& batchNormImageContext = dynamic_cast<BatchNormalizationImageContext&>(imageContext);

        const size_t dataSize = data.GetSize();

        batchNormImageContext.normalized.ReshapeAndResize(data.GetShape());

        const float* input = data.GetData().data();
        float* normalized = batchNormImageContext.normalized.GetData().data();
        float* output = data.GetData().data();

        const float* means = batchMean_.GetData().data();
        const float* inverseStdDev = inverseStdDev_.GetData().data();

        const float* gamma = gamma_.GetData().data();
        const float* beta = beta_.GetData().data();

        for (size_t i = 0; i < dataSize; i++)
        {
            const size_t channel = i % channels_;

            normalized[i] = (input[i] - means[channel]) * inverseStdDev[channel];

            output[i] = gamma[channel] * normalized[i] + beta[channel];
        }
	}

	void BatchNormalization::Backpropagate(Tensor& gradient, WorkerContext& workerContext, ImageContext& imageContext)
	{
        auto& batchNormWorkerContext = dynamic_cast<BatchNormalizationWorkerContext&>(workerContext);
        auto& batchNormImageContext = dynamic_cast<BatchNormalizationImageContext&>(imageContext);

        const size_t dataSize = gradient.GetSize();

        const float* normalized = batchNormImageContext.normalized.GetData().data();

        const float* gamma = gamma_.GetData().data();
        const float* inverseStdDev = inverseStdDev_.GetData().data();

        const float* batchBetaGradients = batchBetaGradients_.GetData().data();
        const float* batchGammaGradients = batchGammaGradients_.GetData().data();

        float* gradients = gradient.GetData().data();

        float* gammaGradients = batchNormWorkerContext.gammaGradients.GetData().data();
        float* betaGradients = batchNormWorkerContext.betaGradients.GetData().data();

        const float totalValues = static_cast<float>(totalBatchValues_);

        for (size_t i = 0; i < dataSize; i++)
        {
            const size_t channel = i % channels_;

            const float dy = gradients[i];
            const float xHat = normalized[i];

            // Accumulate trainable parameter gradients.
            betaGradients[channel] += dy;
            gammaGradients[channel] += dy * xHat;

            // Calculate input gradient.
            gradients[i] = (gamma[channel] * inverseStdDev[channel] / totalValues) *
                (totalValues * dy - batchBetaGradients[channel] - xHat * batchGammaGradients[channel]);
        }
	}

    void BatchNormalization::ApplyGradients(Optimizer& optimizer, int batchSize)
    {
        //mapping the specific layer updates to gamma_ and beta_ parameters
        optimizer.Apply(gamma_, gammaGradients_, batchSize, adamContextGamma_);
        optimizer.Apply(beta_, betaGradients_, batchSize, adamContextBeta_);
    }

	void BatchNormalization::ResetGradients(WorkerContext& workerContext)
	{
        auto& batchNormWorkerContext = dynamic_cast<BatchNormalizationWorkerContext&>(workerContext);

        std::fill(batchNormWorkerContext.gammaGradients.GetData().begin(), batchNormWorkerContext.gammaGradients.GetData().end(), 0.0f);
        std::fill(batchNormWorkerContext.betaGradients.GetData().begin(), batchNormWorkerContext.betaGradients.GetData().end(), 0.0f);

        std::fill(gammaGradients_.GetData().begin(), gammaGradients_.GetData().end(), 0.0f);
        std::fill(betaGradients_.GetData().begin(), betaGradients_.GetData().end(), 0.0f);

        std::fill(batchGammaGradients_.GetData().begin(), batchGammaGradients_.GetData().end(), 0.0f);
        std::fill(batchBetaGradients_.GetData().begin(), batchBetaGradients_.GetData().end(), 0.0f);
	}

	void BatchNormalization::AccumulateGradients(WorkerContext& workerContext)
	{
        auto& batchNormWorkerContext = dynamic_cast<BatchNormalizationWorkerContext&>(workerContext);

        const size_t channels = channels_;

        const float* workerGamma = batchNormWorkerContext.gammaGradients.GetData().data();
        const float* workerBeta = batchNormWorkerContext.betaGradients.GetData().data();

        float* gammaGradients = gammaGradients_.GetData().data();
        float* betaGradients = betaGradients_.GetData().data();

        for (size_t i = 0; i < channels; i++)
        {
            gammaGradients[i] += workerGamma[i];
            betaGradients[i] += workerBeta[i];
        }
	}
}