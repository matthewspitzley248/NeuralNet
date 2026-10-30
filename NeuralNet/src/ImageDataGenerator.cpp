#include <NeuralNet/ImageDataGenerator.h>

namespace NN
{
	ImageDataGenerator::ImageDataGenerator(ImageSettings& imageSettings, GeneratorSettings& generatorSettings, mt19937& randomGenerator, size_t workerCount)
		: imageSettings_(imageSettings),
		generatorSettings_(generatorSettings),
		randomGenerator_(randomGenerator)
	{
		//loop through the directory and grab the different labels (Folders) for the pictures
		for (const auto& labelFolder : filesystem::directory_iterator(generatorSettings.directory))
		{
			//if it is not a folder (collection of images) then continue to the next item
			if (!labelFolder.is_directory())
				continue;

			labels_.push_back(labelFolder.path().filename().string());
		}

		sort(labels_.begin(), labels_.end());

		for (int labelIndex = 0; labelIndex < labels_.size(); labelIndex++)
		{			
			const filesystem::path labelFolder = generatorSettings_.directory / labels_[labelIndex];

			for (const auto& imageFile : filesystem::directory_iterator(labelFolder))
			{
				//if it is a normal file add it to the label set
				if (imageFile.is_regular_file())
				{
					LabeledImagePath labeledImage;
					labeledImage.labelIndex = labelIndex;
					labeledImage.imagePath = imageFile.path();

					labeledImages_.push_back(move(labeledImage));
				}
			}
		}

		if (generatorSettings_.shuffle)
		{
			shuffle(labeledImages_.begin(),	labeledImages_.end(), randomGenerator_);
		}

		//Start the workers for the gen
		workerPool_.Start(workerCount);

		// Start the background thread that fills the preloaded batch queue.
		batchLoaderThread_ = thread(&ImageDataGenerator::BatchLoaderLoop, this);
	}

	ImageDataGenerator::~ImageDataGenerator()
	{
		{
			std::lock_guard<std::mutex> lock(batchQueueMutex_);
			stopBatchLoader_ = true;
		}

		batchQueueCondition_.notify_all();

		if (batchLoaderThread_.joinable())
		{
			batchLoaderThread_.join();
		}

		workerPool_.Stop();
	}

	void ImageDataGenerator::NewEpoch()
	{
		unique_lock<mutex> lock(batchQueueMutex_);

		//Wait until the previous epoch has finished loading and all its batches have been consumed
		batchQueueCondition_.wait(lock, [this]()
			{
				return stopBatchLoader_
					|| (!epochActive_ && epochProductionFinished_ && preloadedBatches_.empty() && !loadingBatch_);
			});

		if (stopBatchLoader_)
			return;

		// Shuffle the images if requested.
		if (generatorSettings_.shuffle)
		{
			std::shuffle(labeledImages_.begin(), labeledImages_.end(), randomGenerator_);
		}

		// Atomic Tracking Update: Safe lock-free indexing reset
		currentImageIndex_.store(0, std::memory_order_release);

		//Activate the new epoch and allow the producer to start loading
		epochProductionFinished_ = false;
		epochActive_ = true;

		lock.unlock();
		batchQueueCondition_.notify_all();
	}

	ImageBatch ImageDataGenerator::NextBatch()
	{
		unique_lock<mutex> lock(batchQueueMutex_);

		batchQueueCondition_.wait(lock, [this]()
			{
				return !preloadedBatches_.empty() || epochProductionFinished_ || stopBatchLoader_ || batchLoaderException_ != nullptr;
			});

		//Propagate errors from the background loading thread
		if (batchLoaderException_)
		{
			exception_ptr exception = batchLoaderException_;
			batchLoaderException_ = nullptr;

			lock.unlock();
			rethrow_exception(exception);
		}

		//No more batches are available for this epoch
		if (preloadedBatches_.empty())
		{
			return ImageBatch{};
		}

		//Consume the oldest completed batch
		ImageBatch batch = move(preloadedBatches_.front());
		preloadedBatches_.pop();

		//Let the producer know it can preload another batch
		lock.unlock();
		batchQueueCondition_.notify_all();

		return batch;
	}


	int ImageDataGenerator::GetBatchSize()
	{
		return generatorSettings_.batchSize;
	}

	int ImageDataGenerator::GetBatchCount()
	{
		if (labeledImages_.empty()) 
			return 0;

		return static_cast<int>((labeledImages_.size() - 1) / generatorSettings_.batchSize) + 1;
	}

	int ImageDataGenerator::GetLabelSize()
	{
		return labels_.size();
	}


	void ImageDataGenerator::BatchLoaderLoop()
	{
		while (true)
		{
			{
				unique_lock<mutex> lock(batchQueueMutex_);

				batchQueueCondition_.wait(lock, [this]()
					{
						return stopBatchLoader_ ||
							(epochActive_ && !epochProductionFinished_ && !loadingBatch_ && preloadedBatches_.size() < maxPreloadedBatches_);
					});

				if (stopBatchLoader_)
					return;

				//Mark that the producer is preparing a batch
				loadingBatch_ = true;
			}

			//Load outside the queue mutex so NextBatch() can consume already-preloaded batches while this batch is loading
			ImageBatch batch;

			try
			{
				batch = LoadBatch();
			}
			catch (...)
			{
				lock_guard<mutex> lock(batchQueueMutex_);

				loadingBatch_ = false;
				epochProductionFinished_ = true;
				epochActive_ = false;

				batchQueueCondition_.notify_all();

				//Preserve the exception for the consumer to handle
				batchLoaderException_ = current_exception();
				continue;
			}

			{
				lock_guard<mutex> lock(batchQueueMutex_);

				//If shutdown was requested while loading, discard this batch
				if (stopBatchLoader_)
				{
					loadingBatch_ = false;
					batchQueueCondition_.notify_all();
					return;
				}

				//An empty batch means there are no more images this epoch
				if (batch.imageLabels.empty())
				{
					epochProductionFinished_ = true;
					epochActive_ = false;
				}
				else
				{
					preloadedBatches_.push(move(batch));
				}

				loadingBatch_ = false;
			}

			//Wake NextBatch() and allow the producer to check queue capacity
			batchQueueCondition_.notify_all();
		}
	}

	ImageBatch ImageDataGenerator::LoadBatch()
	{
		ImageBatch batch;
		const size_t totalImages = labeledImages_.size();

		// 1. Lock-Free Index Acquisition Phase
		// Atomic index increments eliminate the global mutex check entirely.
		size_t startIndex = currentImageIndex_.load(std::memory_order_relaxed);
		size_t batchCount = 0;

		while (startIndex < totalImages)
		{
			size_t remainingImages = totalImages - startIndex;
			batchCount = std::min(static_cast<size_t>(generatorSettings_.batchSize), remainingImages);

			if (batchCount == 0) break;

			if (currentImageIndex_.compare_exchange_weak(startIndex, startIndex + batchCount,
				std::memory_order_seq_cst,
				std::memory_order_relaxed))
			{
				break; // Successfully reserved batch segment boundaries safely
			}
		}

		if (batchCount == 0)
			return batch;

		batch.imageLabels.resize(batchCount);

		// 2. Coarse-Grained Task Distribution Architecture
		// Instead of one job per single image, divide the work evenly among available threads.
		const size_t numWorkers = workerPool_.GetWorkerCount();
		const size_t numJobs = std::min(numWorkers, batchCount);

		vector<WorkerPool::Job> jobs;
		jobs.reserve(numJobs);

		for (size_t jobIdx = 0; jobIdx < numJobs; ++jobIdx)
		{
			// Precalculate localized slice ranges for individual thread assignments
			size_t imgStart = (jobIdx * batchCount) / numJobs;
			size_t imgEnd = ((jobIdx + 1) * batchCount) / numJobs;

			jobs.emplace_back([this, &batch, startIndex, imgStart, imgEnd](size_t)
				{
					// Threads process their segment sequentially.
					// This triggers hard-drive streaming optimizations.
					for (size_t i = imgStart; i < imgEnd; ++i)
					{
						const size_t globalIdx = startIndex + i;
						batch.imageLabels[i].labelIndex = labeledImages_[globalIdx].labelIndex;

						// Zero Allocation String View Pass: Avoid copying path objects entirely
						batch.imageLabels[i].image = imgLoader_.Load(labeledImages_[globalIdx].imagePath.string());
					}
				});
		}

		workerPool_.SubmitJobs(std::move(jobs));
		workerPool_.WaitForJobs();

		return batch;
	}
}