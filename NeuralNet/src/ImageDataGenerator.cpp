#include <NeuralNet/ImageDataGenerator.h>

namespace NN
{
	ImageDataGenerator::ImageDataGenerator(ImageSettings& imageSettings, GeneratorSettings& generatorSettings, mt19937& randomGenerator)
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

			labels.push_back(labelFolder.path().filename().string());
		}

		sort(labels.begin(), labels.end());

		for (int labelIndex = 0; labelIndex < labels.size(); labelIndex++)
		{			
			const filesystem::path labelFolder = generatorSettings_.directory / labels[labelIndex];

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
	}

	void ImageDataGenerator::NewEpoch()
	{
		currentImageIndex_ = 0;

		if (generatorSettings_.shuffle)
		{
			std::shuffle(
				labeledImages_.begin(),
				labeledImages_.end(),
				randomGenerator_);
		}
	}

	ImageBatch ImageDataGenerator::NextBatch()
	{
		ImageBatch batch;
		//Create a batch until BatchSize is reached or it hits the end of the labeledImages
		for (int i = 0; i < generatorSettings_.batchSize && currentImageIndex_ < labeledImages_.size(); i++)
		{
			const LabeledImagePath& labeledImage = labeledImages_[currentImageIndex_];

			ImageLabel imglabel;
			imglabel.labelIndex = labeledImage.labelIndex;
			imglabel.image = imgLoader.Load(labeledImage.imagePath);


			batch.imageLabels.push_back(move(imglabel));

			currentImageIndex_++;
		}

		return batch;
	}


	int ImageDataGenerator::GetBatchSize()
	{
		return generatorSettings_.batchSize;
	}
}