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
            var mlContext = new MLContext();
            IDataView data = mlContext.Data.LoadFromTextFile<IrisData>(_dataPath, hasHeader:false, separatorChar: ',');
            

            //create learning pipeline
            string featuresColumnName = "Features";
            var pipeline = mlContext.Transforms
                                .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, clustersCount: 3));
            // var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
            //     .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
            //     .AppendCacheCheckpoint(mlContext)
            //     .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumnName: "Label", featureColumnName: "Features"))
            //     .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            
            //train the model
            var model  = pipeline.Fit(data);

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
