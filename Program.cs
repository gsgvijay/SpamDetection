namespace SpamDetection
{
    using Microsoft.ML;
    using Microsoft.ML.Core.Data;
    using Microsoft.ML.Runtime;
    using Microsoft.ML.Runtime.Data;
    using SpamDetection.DataStructures;

    using System;
    using System.ComponentModel.Composition;
    using System.IO;
    using System.IO.Compression;
    using System.Linq;
    using System.Net;

    class Program
    {
        private static string AppPath =>
            Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        
        private static string DataDirectoryPath =>
            Path.Combine(Program.AppPath, "Data", "spamFolder");
        
        private static string TrainDataPath =>
            Path.Combine(Program.AppPath, "Data", "spamFolder", "SMSSpamCollection");

        public class MyInput
        {
            public string Label { get; set; }
        }

        public class MyOutput
        {
            public bool Label { get; set; }
        }

        public class MyLambda
        {
            [Export("MyLambda")]
            public ITransformer MyTransformer =>
                ML.Transforms.CustomMappingTransformer<MyInput, MyOutput>(MyAction, "MyLambda");
            
            [Import]
            public MLContext ML { get; set; }

            public static void MyAction(MyInput input, MyOutput output)
            {
                output.Label = input.Label == "spam";
            }
        }

        static void Main(string[] args)
        {
            if (false == File.Exists(Program.TrainDataPath))
            {
                using (var client = new WebClient())
                {
                    client.DownloadFile(@"https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip", "spam.zip");
                }

                ZipFile.ExtractToDirectory("spam.zip", Program.DataDirectoryPath);
            }

            var context = new MLContext();
            
            var reader = new TextLoader(context, new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.Text, 0),
                    new TextLoader.Column("Message", DataKind.Text, 1)
                }
            });

            var data = reader.Read(new MultiFileSource(Program.TrainDataPath));

            var estimator = context.Transforms.CustomMapping<MyInput, MyOutput>(MyLambda.MyAction, "MyLambda")
                .Append(context.Transforms.Text.FeaturizeText("Message", "Features"))
                .Append(context.BinaryClassification.Trainers.StochasticDualCoordinateAscent());
            
            var cvResult = context.BinaryClassification.CrossValidate(data, estimator, numFolds: 5);
            var aucs = cvResult.Select(r => r.metrics.Auc);
            Console.WriteLine($"The AUC is {aucs.Average()}");

            var model = estimator.Fit(data);
            var inPipe = new TransformerChain<ITransformer>(model.Take(model.Count() - 1).ToArray());

            var lastTransformer = new BinaryPredictionTransformer<IPredictorProducing<float>>(
                context,
                model.LastTransformer.Model,
                inPipe.GetOutputSchema(data.Schema),
                model.LastTransformer.FeatureColumn,
                threshold: 0.15f,
                thresholdColumn: DefaultColumnNames.Probability);
            var parts = model.ToArray();
            parts[parts.Length - 1] = lastTransformer;
            var newModel = new TransformerChain<ITransformer>(parts);
            var predictor = newModel.MakePredictionFunction<Input, Prediction>(context);

            Program.ClassifyMessage(predictor, "That's a great idea. It should work.");
            Program.ClassifyMessage(predictor, "Free medicine winner! Congratulations");
            Program.ClassifyMessage(predictor, "Yes we should meet over the weekend");
            Program.ClassifyMessage(predictor, "You win pills and free entry vouchers");
        }

        public static void ClassifyMessage(PredictionFunction<Input, Prediction> predictor, string message)
        {
            var input = new Input { Message = message };
            var prediction = predictor.Predict(input);
            var output = prediction.IsSpam
                ? "Spam"
                : "Not Spam";

            Console.WriteLine($"The message '{input.Message}' is '{output}'");
        }
    }
}
