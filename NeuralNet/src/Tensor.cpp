#include <NeuralNet/Tensor.h>

namespace NN 
{
	/// <summary>
	/// creates a new tensor object of the given shape
	/// </summary>
	/// <param name="shape">vector<int> - shape of the new tensor object</param>
	Tensor::Tensor(std::vector<size_t> shape)
		: shape_(shape)
	{
		size_t totalElements = std::accumulate(
			shape_.begin(),
			shape_.end(),
			size_t{ 1 },
			std::multiplies<size_t>()
		);
		
		data_.resize(totalElements);
	}

	Tensor& Tensor::operator=(Tensor&& other) noexcept
	{
		// 1. Self-assignment check (safeguards against: a = std::move(a);)
		if (this != &other)
		{
			// 2. Perform the move inside the function body
			shape_ = std::move(other.shape_);
			data_ = std::move(other.data_);
		}

		return *this;
	}

	//float& Tensor::At(const std::vector<size_t>& index)
	//{
	//	//check for bounds of shape
	//	if (index.size() != shape_.size())
	//	{
	//		printf("[Tensor] Out of bounds! Size: %zu, Shape: \n", data_.size());
	//		throw std::out_of_range("Tensor index out of bounds.");
	//	}

	//	int returnIndex = 0;
	//	int stride = 1;
	//	for (int i = 0; i < index.size(); i++)
	//	{
	//		//say want (300, 300, 3) would be At(299, 299, 2), shape would be (300, 300, 3)
	//		if (index.at(i) < 0 || index.at(i) >= shape_.at(i))
	//		{
	//			printf("[Tensor] Out of bounds! Size: %zu, Shape: \n", data_.size());
	//			throw std::out_of_range("Tensor index out of bounds.");
	//		}

	//		returnIndex += index.at(i) * stride;
	//		stride *= shape_.at(i);
	//	}

	//	return data_.at(returnIndex);
	//}

	///*float& Tensor::At(const int y, const int x, const int z)
	//{
	//	return data_.at(y + (x * 3) + (z * 9));
	//}*/

	//float& Tensor::At(size_t index)
	//{
	//	if (shape_.size() != 1)
	//	{
	//		throw std::runtime_error("Tensor is not 1D");
	//	}

	//	if (index < 0 || index >= shape_[0])
	//	{
	//		printf("[Tensor] Out of bounds! Index: %zu, Size: %zu, Shape: \n", index, data_.size());
	//		throw std::out_of_range("Tensor index out of bounds");
	//	}

	//	return data_[index];
	//}

	float& Tensor::AtFlat(size_t index)
	{
		if (index >= data_.size())
		{
			printf("[Tensor] Out of bounds! Index: %zu, Size: %zu\n", index, data_.size());

			throw std::out_of_range("Tensor index out of bounds");
		}

		return data_.at(index);
	}

	//const float& Tensor::At(const std::vector<size_t>& index) const
	//{
	//	return const_cast<Tensor*>(this)->At(index);
	//}

	///*const float& Tensor::At(const int y, const int x, const int z) const
	//{
	//	return const_cast<Tensor*>(this)->At(y, x, z);
	//}*/

	//const float& Tensor::At(size_t index) const
	//{
	//	return const_cast<Tensor*>(this)->At(index);
	//}

	const float& Tensor::AtFlat(size_t index) const
	{
		return const_cast<Tensor*>(this)->AtFlat(index);
	}

	const std::vector<size_t>& Tensor::GetShape() const
	{
		return shape_;
	}

	const size_t& Tensor::GetSize() const
	{
		return data_.size();
	}

	std::vector<float>& Tensor::GetData()
	{
		return data_;
	}

	const std::vector<float>& Tensor::GetData() const
	{
		return data_;
	}


	void Tensor::SetShape(const std::vector<size_t>& newShape)
	{
		//Check the size of new shape to match
		size_t newSize = 1;

		for (int dimension : newShape)
		{
			if (dimension <= 0)
			{
				throw std::invalid_argument(
					"Tensor dimensions must be positive."
				);
			}

			newSize *= static_cast<size_t>(dimension);
		}

		if (newSize != data_.size())
		{
			throw std::invalid_argument(
				"New shape must contain the same number of elements."
			);
		}

		shape_ = newShape;
	}

	void Tensor::ReshapeAndResize(const std::vector<size_t>& newShape)
	{
		size_t newSize = 1;
		for (size_t dimension : newShape)
		{
			if (dimension <= 0)
			{
				throw std::invalid_argument("Tensor dimensions must be positive.");
			}
			newSize *= static_cast<size_t>(dimension);
		}

		// This is the performance win: std::vector::resize() will NOT allocate 
		// fresh heap memory if newSize is less than or equal to the vector's current capacity().
		data_.resize(newSize);
		shape_ = newShape;
	}
}