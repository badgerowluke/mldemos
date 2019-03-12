using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace sentiment_demo
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            MLContext mlContext = new MLContext();
            TrainCatalogBase.TrainTestData splitDataView = LoadData(mlContext);
            ITransformer model = BuildAndTrain(mlContext, splitDataView.TrainSet);
            Evaluate(mlContext, model, splitDataView.TestSet);
            SaveModelAsFile(mlContext, model);
            Predict(mlContext, model);
            PredictWithModelLoadedFromFile(mlContext);

            
        }
        public static TrainCatalogBase.TrainTestData LoadData(MLContext context)
        {
            IDataView dataView = context.Data.LoadFromTextFile<SentimentData>(_dataPath,hasHeader:false);
            TrainCatalogBase.TrainTestData splitDataView = context.BinaryClassification.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }
        static ITransformer BuildAndTrain(MLContext context, IDataView trainSet)
        {
            Console.WriteLine("=============== Create and Train the Model ===============");
            var pipeline = context.Transforms.Text.FeaturizeText(outputColumnName: DefaultColumnNames.Features, inputColumnName: nameof(SentimentData.SentimentText))
                        .Append(context.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));

            var model = pipeline.Fit(trainSet);
            
            Console.WriteLine("=============== End of Training ===============");
            return model;                                                                                                                                                                                                                                                                                                                                  
        }
        static void Evaluate(MLContext context, ITransformer model, IDataView testSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            var predictions = model.Transform(testSet);
            var metrics = context.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
            
        }
        private static void SaveModelAsFile(MLContext context, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                context.Model.Save(model, fs);
            }
        }
        private static void Predict(MLContext context, ITransformer model)
        {
            var predictFunction = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(context);
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText ="This is a very bad steak"
            };
            var resultPrediction = predictFunction.Predict(sampleStatement);
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {sampleStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }
        private static void PredictWithModelLoadedFromFile(MLContext context)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti."
                }
            };

            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = context.Model.Load(stream);
            }   
            // Create prediction engine
            var sentimentStreamingDataView = context.Data.LoadFromEnumerable(sentiments);
            var predictions = loadedModel.Transform(sentimentStreamingDataView);

            // Use the model to predict whether comment data is toxic (1) or nice (0).
            var predictedResults = context.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: true);        
            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");    
            var sentimentsAndPredictions = sentiments.Zip(predictedResults, (sentiment, prediction) => (sentiment, prediction));
            foreach (var item in sentimentsAndPredictions)
            {
                
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(Convert.ToBoolean(item.prediction.Prediction) ? "Positive" : "Negative")} | Probability: {item.prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}
