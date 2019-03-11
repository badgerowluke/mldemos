using Microsoft.ML.Data;
namespace clustering_demo
{
    public class IrisData
    {
        [LoadColumn(0)]
        public float SepalLength;

        [LoadColumn(1)]
        public float SepalWidth;

        [LoadColumn(2)]
        public float PetalLength;

        [LoadColumn(3)]
        public float PetalWidth;
    }

    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distances;
    }
    static class TestIrisData
    {
        internal static readonly IrisData Setosa = new IrisData{
            SepalLength = 5.1f,
            SepalWidth = 3.5f,
            PetalLength = 1.4f,
            PetalWidth = 0.2f
        };
    }

}