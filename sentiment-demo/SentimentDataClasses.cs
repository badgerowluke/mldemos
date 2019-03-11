using Microsoft.ML.Data;

namespace sentiment_demo
{
    public class SentimentData
    {
        [Column(ordinal: "0", name: "Label")]        
        public float Sentiment;
        [Column(ordinal: "1")]
        public string SentimentText;
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictionLabel")]
        public bool Prediction { get; set; }

        [ColumnName("Perovibility")]
        public float Probability { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }
        
    }
}