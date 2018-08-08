using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;
using UnityEngine;


public class RecognizedObject
{
    public string Label;
    public float Score;

    Rect _rectangle;

    public RecognizedObject(float x0, float y0, float x1, float y1, string label, float score)
    {
        _rectangle = new Rect(x0, y0, x1 - x0, y1 - y0);
        Label = label;
        Score = score;
    }

    public Rect ScreenRectangle()
    {
        int w = Screen.width;
        int h = Screen.height;
        return new Rect(_rectangle.x * w, _rectangle.y * h, _rectangle.width * w, _rectangle.height * h);
    }

    public override string ToString()
    {
        double percent = Score * 100;
        return $"<b>{Label}:</b> {percent:0.0} %";
    }
}

class ObjectRecognizer : IDisposable
{
    const int _imageSize = 224;
    const int _imageMean = 0;
    const float _imageScale = 1;
    const float _minScore = 0.3f;

    Dictionary<int, CatalogItem> _catalog;
    private TFGraph _graph;

    public ObjectRecognizer(byte[] model, string labels)
    {
#if UNITY_ANDROID
            TensorFlowSharp.Android.NativeBinding.Init();
#endif
        _catalog = CatalogUtil.ReadCatalogItems(labels).ToDictionary(l => l.Id);
        _graph = new TFGraph();
        _graph.Import(new TFBuffer(model));
    }

    public static long Delay = 0;

    public Task<IEnumerable<RecognizedObject>> DetectAsync(Texture2D texture)
    {
        var pixels = GetPixels(texture);

        return Task.Run(() =>
        {
            using (var session = new TFSession(_graph))
            using (var tensor = TransformInput(pixels))
            {
                var runner = session.GetRunner();
                runner.AddInput(_graph["image_tensor"][0], tensor)
                      .Fetch(_graph["detection_boxes"][0],
                             _graph["detection_scores"][0],
                             _graph["detection_classes"][0]);
                //  _graph["num_detections"][0]);

                var watch = System.Diagnostics.Stopwatch.StartNew();
                var output = runner.Run();

                Delay = watch.ElapsedMilliseconds;

                var boxes = (float[,,])output[0].GetValue(jagged: false);
                var scores = (float[,])output[1].GetValue(jagged: false);
                var classes = (float[,])output[2].GetValue(jagged: false);

                foreach (var ts in output)
                {
                    ts.Dispose();
                }

                return GetRecognizedObjects(boxes, scores, classes);
            }
        });
    }

    public TFTensor TransformInput(Color32[] pixels)
    {
        byte[] floatValues = new byte[_imageSize * _imageSize * 3];

        for (int i = 0; i < pixels.Length; ++i)
        {
            var color = pixels[i];

            floatValues[i * 3 + 0] = (byte)((color.r - _imageMean) / _imageScale);
            floatValues[i * 3 + 1] = (byte)((color.g - _imageMean) / _imageScale);
            floatValues[i * 3 + 2] = (byte)((color.b - _imageMean) / _imageScale);
        }

        var shape = new TFShape(1, _imageSize, _imageSize, 3);
        return TFTensor.FromBuffer(shape, floatValues, 0, floatValues.Length);
    }

    private IEnumerable<RecognizedObject> GetRecognizedObjects(float[,,] boxes, float[,] scores, float[,] classes)
    {
        var x = boxes.GetLength(0);
        var y = boxes.GetLength(1);
        var z = boxes.GetLength(2);

        float ymin = 0, xmin = 0, ymax = 0, xmax = 0;

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                float score = scores[i, j];
                if (score < _minScore) continue;

                for (int k = 0; k < z; k++)
                {
                    var box = boxes[i, j, k];
                    switch (k)
                    {
                        case 0:
                            ymin = box;
                            break;
                        case 1:
                            xmin = box;
                            break;
                        case 2:
                            ymax = box;
                            break;
                        case 3:
                            xmax = box;
                            break;
                    }
                }

                int id = Convert.ToInt32(classes[i, j]);
                CatalogItem item;

                if (_catalog.TryGetValue(id, out item))
                    yield return new RecognizedObject(xmin, ymin, xmax, ymax, item.DisplayName, score);
            }
        }
    }

    public Color32[] GetPixels(Texture2D tex)
    {
        Rect texR = new Rect(0, 0, _imageSize, _imageSize);
        RenderTexture rtt = new RenderTexture(_imageSize, _imageSize, 32);
        Graphics.SetRenderTarget(rtt);
        GL.LoadPixelMatrix(0, 1, 1, 0);
        GL.Clear(true, true, new Color(0, 0, 0, 0));
        Graphics.DrawTexture(new Rect(0, 0, 1, 1), tex);

        tex.Resize(_imageSize, _imageSize);
        tex.ReadPixels(texR, 0, 0, true);
        //  tex.Apply(true);

        var pixels = tex.GetPixels32();
        Array.Reverse(pixels);
        return pixels;
    }

    public void Dispose()
    {
        _graph.Dispose();
    }
}
