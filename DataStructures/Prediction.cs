namespace SpamDetection.DataStructures
{
    using Microsoft.ML.Runtime.Api;
    public class Prediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsSpam { get; set; }

        public float Score { get; set; }

        public float Probability { get; set; }
    }
}