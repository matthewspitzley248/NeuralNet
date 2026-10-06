#include <NeuralNet/Tensor.h>

namespace NN 
{
	/// <summary>
	/// creates a new tensor object of the given shape
	/// </summary>
	/// <param name="shape">vector<int> - shape of the new tensor object</param>
	Tensor::Tensor(std::vector<int> shape)
	{
		shape_ = shape;

		int dataPoints = 1;
		while (!shape.empty())
		{
			dataPoints *= shape.back();
			shape.pop_back();
		}

		data_.resize(dataPoints);
	}

	/// <summary>
	/// gets teh value at the given location index.
	/// </summary>
	/// <param name="index">vector<int> - location to get value</param>
	/// <returns>float - reference to value at given index</returns>
	float& Tensor::At(const std::vector<int>& index)
	{
		//check for bounds of shape
		if (index.size() != shape_.size())
		{
			throw std::out_of_range("Tensor index out of bounds.");
		}

		int returnIndex = 0;
		int stride = 1;
		for (int i = 0; i < index.size(); i++)
		{
			//say want (300, 300, 3) would be At(299, 299, 2), shape would be (300, 300, 3)
			if (index.at(i) < 0 || index.at(i) >= shape_.at(i))
			{
				throw std::out_of_range("Tensor index out of bounds.");
			}

			returnIndex = index.at(i) * stride;
			stride *= shape_.at(i);
		}

		return data_.at(returnIndex);
	}

	float& Tensor::At(const int y, const int x, const int z)
	{
		return data_.at(y + (x * 3) + (z * 9));
	}

	/// <summary>
	/// returns the shape of the tensor
	/// </summary>
	/// <returns>vector<int> - returns a constant reference to the shape of the tensor</returns>
	const std::vector<int>& Tensor::GetShape() const
	{
		return shape_;
	}

	const size_t& Tensor::GetSize() const
	{
		return data_.size();
	}
}