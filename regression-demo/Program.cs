using System;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace regression_demo
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");        
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            //the random seed here is for reteable/determiistic results accros multiple trainings
            var context = new MLContext(seed:0);
            var model = Train(context: context, dataPath: _trainDataPath);
            Evaluate(context, model);
            TestSinglePrediction(context);
        }
        static ITransformer Train(MLContext context, string dataPath)
        {
            var dataView = context.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader:true, separatorChar: ',');
            var pipeline = context.Transforms.CopyColumns(outputColumnName:"Label", inputColumnName:"FareAmount")
                            .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName:"VendorIdEncoded", inputColumnName:"VendorId"))
                            .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName:"RateCodeEncoded", inputColumnName:"RateCode"))
                            .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName:"PaymentTypeEncoded", inputColumnName:"PaymentType"))
                            .Append(context.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripTime", "TripDistance", "PaymentTypeEncoded"))
                            .Append(context.Regression.Trainers.FastTree());
            var model= pipeline.Fit(dataView);   
            SaveModelAsFile(context, model);         
            return model;
        }
        private static void SaveModelAsFile(MLContext context, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                context.Model.Save(model, fs);
            }
        }
        static void Evaluate(MLContext context, ITransformer model)
        {
            var dataView = context.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader:true, separatorChar: ',');
            var predictions = model.Transform(dataView);
            var metrics = context.Regression.Evaluate(predictions, "Label","Score");
            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}");
        }
        static void TestSinglePrediction(MLContext context)
        {
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = context.Model.Load(stream);
            }   
            var predictionFunction = loadedModel.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(context);
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };
            var prediction = predictionFunction.Predict(taxiTripSample);
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");

        }
    }
}
