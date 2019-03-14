using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;



namespace multiclass_demo
{
    class Program
    {
        static readonly string _trainDataPath =  Path.Combine(Environment.CurrentDirectory, "Data", "issues_train.tsv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "issues_test.tsv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "model.zip");

        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;
        static MLContext context;
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            //the random seed here is for reteable/determiistic results accros multiple trainings
            context = new MLContext(seed: 0);
            _trainingDataView = context.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader:true);

            var trainingPipeline = BuildAndTrainModel(_trainingDataView);
            Evaulate();
            SaveModelAsFile( _trainedModel);
            PredictIssue();
        }

        static EstimatorChain<KeyToValueMappingTransformer> BuildAndTrainModel(IDataView trainingData)
        {
            var pipeline = context.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
            .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
            .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
            .Append(context.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
            .AppendCacheCheckpoint(context);

            var trainingPipeline = pipeline.Append(context.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(DefaultColumnNames.Label, DefaultColumnNames.Features))
                                            .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            _trainedModel = trainingPipeline.Fit(trainingData);
            _predEngine = _trainedModel.CreatePredictionEngine<GitHubIssue, IssuePrediction>(context);
            
            GitHubIssue issue = new GitHubIssue() {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var prediction = _predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
            return trainingPipeline;

        }
        static void Evaulate()
        {
            var testDataView = context.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader:true);
            var testMetrics = context.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.AccuracyMicro:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.AccuracyMacro:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");
            
        }
        private static void SaveModelAsFile( ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                context.Model.Save(model, fs);
            }
        } 
        static void PredictIssue()
        {
            ITransformer loadedModel;
            using(var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = context.Model.Load(stream);
                GitHubIssue singleIssue = new GitHubIssue() 
                { 
                    Title = "Entity Framework crashes", 
                    Description = "When connecting to the database, EF is crashing" 
                };
                _predEngine = loadedModel.CreatePredictionEngine<GitHubIssue, IssuePrediction>(context);
                var prediction = _predEngine.Predict(singleIssue);
                Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
            }
        }       
    }
}
