using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace sentiment_demo
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-data.tsv");
        static readonly string _testDataPath= Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-test.tsv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static TextLoader _textLoader;
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            MLContext mlContext = new MLContext(seed: 0);
            _textLoader = mlContext.Data.CreateTextLoader(
                columns: new TextLoader.Column[]
                {
                    new TextLoader.Column("Label", DataKind.Bool, 0),
                    new TextLoader.Column("SentimentText", DataKind.Text,1)
                },
                separatorChar: '\t',
                hasHeader: true
            );
            var model = Train(context: mlContext, trainingPath: _trainDataPath);
            Evaluate(context: mlContext, model: model);
            SaveModelAsFile(context: mlContext, model: model);
            Predict(context: mlContext, model: model);
            // PredictWithModelLoadedFromFile(context: mlContext);
            
        }
        static ITransformer Train(MLContext context, string trainingPath)
        {
            Console.WriteLine("=============== Create and Train the Model ===============");
            IDataView dataView = _textLoader.Read(trainingPath);
            var pipeline = context.Transforms.Text
                        .FeaturizeText(inputColumnName: "SentimentText", outputColumnName: "Features")
                        .Append(context.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));

            var model = pipeline.Fit(dataView);
            
            Console.WriteLine("=============== End of Training ===============");
            return model;                                                                                                                                                                                                                                                                                                                                  
        }
        static void Evaluate(MLContext context, ITransformer model)
        {
            var dataView = _textLoader.Read(_testDataPath);
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            var predictions = model.Transform(dataView);
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
                SentimentText ="you are a fucking rude asshole"
            };
            var resultPrediction = predictFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {sampleStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Toxic" : "Not Toxic")} | Probability: {resultPrediction.Probability} ");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }
        private static void PredictWithModelLoadedFromFile(MLContext context)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This is a very rude movie"
                },
                new SentimentData
                {
                    SentimentText = "I love this article."
                }
            };
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = context.Model.Load(stream);
            }   
            // Create prediction engine
            var sentimentStreamingDataView = context.Data.ReadFromEnumerable(sentiments);
            var predictions = loadedModel.Transform(sentimentStreamingDataView);

            // Use the model to predict whether comment data is toxic (1) or nice (0).
            var predictedResults = context.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);        
            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");    
            var sentimentsAndPredictions = sentiments.Zip(predictedResults, (sentiment, prediction) => (sentiment, prediction));
            foreach (var item in sentimentsAndPredictions)
            {
                
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(Convert.ToBoolean(item.prediction.Prediction) ? "Toxic" : "Not Toxic")} | Probability: {item.prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}
