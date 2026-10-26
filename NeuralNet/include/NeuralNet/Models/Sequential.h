#pragma once

#include <NeuralNet/Models/Model.h>
#include <iostream>

using namespace std;

namespace NN::Models
{
    struct WorkerContext
    {
        vector<unique_ptr<Layers::LayerContext>> layerContexts;
    };

    class Sequential : public Model
    {
    public:
        Sequential(vector<size_t> inputShape, float learningRate = 0.001f);
        Sequential(vector<size_t> inputShape, unsigned int seed, float learningRate = 0.001f);
        Sequential(vector<size_t> inputShape, mt19937& generator, float learningRate = 0.001f);

        template<typename T, typename... Args>
        void Add(Args&&... args)
        {
            layers_.push_back(make_unique<T>(forward<Args>(args)...));
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="loss"></param>
        /// <param name="optimizer"></param>
        /// <param name="metrics"></param>
        void Compile(LossType lossType, OptimizerType optimizerType, vector<Metric> metrics, int threadCount = 8);

        void Fit(int epochs, ImageDataGenerator& trainData, ImageDataGenerator& testData);

    protected:
        Tensor Run(const Tensor& input, WorkerContext& worker);
        Tensor Backpropagation(Tensor gradient, WorkerContext& worker);
        void ApplyGradients(int batchSize);

        float Evaluate(ImageDataGenerator& testData);

        void CreateWorkers(int workerCount);
        void ResetWorkerGradients();
        void AccumulateWorkerGradients();

        //Optimizer optimizer_;
        float learningRate_;

        vector<WorkerContext> workers_;

        //Thread pool shared by Fit() and Evaluate().
        WorkerPool workerPool_;
    };
}