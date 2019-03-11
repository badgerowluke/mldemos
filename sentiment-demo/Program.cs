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
            
        }
        static ITransformer Train(MLContext context, string trainingPath)
        {
            IDataView dataView = _textLoader.Read(trainingPath);
            var pipeline = context.Transforms.Text
                        .FeaturizeText(inputColumnName: "SentimentText", outputColumnName: "Features")
                        .Append(context.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));

            var model = pipeline.Fit(dataView);
            
            return model;                                                                                                                                                                                                                                                                                                                                  
        }
        static void Evaluate(MLContext context, ITransformer model)
        {
            var dataView = _textLoader.Read(_testDataPath);
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
    }
}
