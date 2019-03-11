using System;
using System.IO;

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Data.DataView;

namespace clustering_demo
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");
        
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed:0);
            //set up data load
            TextLoader textLoader = mlContext.Data.CreateTextLoader<IrisData>(false,',');
            var dataView = textLoader.Read(_dataPath);

            //create learning pipeline
            string featuresColumnName = "Features";
            var pipeline = mlContext.Transforms
                                .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, clustersCount: 3));

            
            //train the model
            var model  = pipeline.Fit(dataView);

            //save the model
            using(var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, fs);
            }

            //make predictions
            var predictor = model.CreatePredictionEngine<IrisData, ClusterPrediction>(mlContext);
            var prediction= predictor.Predict(TestIrisData.Setosa);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
            Console.ReadLine();


        }
    }
}
