using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

// Based on https://github.com/Syn-McJ/TFClassify-Unity and https://github.com/migueldeicaza/TensorFlowSharp/tree/master/Examples/ExampleObjectDetection

public class RecognizerController : MonoBehaviour
{
    [SerializeField] TextAsset _model;
    [SerializeField] TextAsset _labels;
    [SerializeField] RawImage _canvas;
    [SerializeField] GUISkin _skin;

    WebCamTexture _webCam;
    ObjectRecognizer _recognizer;
    List<RecognizedObject> _recognizedObjects = new List<RecognizedObject>();
    long _updateTime;

    void Start()
    {
        _recognizer = new ObjectRecognizer(_model.bytes, _labels.text);
        SetupWebCam();
        RecognitionLoop();
    }

    void OnGUI()
    {
        GUI.skin = _skin;

        foreach (var recognizedObject in _recognizedObjects)
        {
            GUI.Box(recognizedObject.ScreenRectangle(), recognizedObject.ToString());
        }

        GUI.Label(new Rect(20, 20, 400, 40), $"Update time: <b>{ObjectRecognizer.Delay}</b> ms");
    }

    void OnApplicationQuit()
    {
        _recognizer.Dispose();
    }

    void SetupWebCam()
    {
        WebCamDevice[] devices = WebCamTexture.devices;

        if (devices.Length == 0)
        {
            Debug.Log("No camera detected");
            return;
        }

        _webCam = new WebCamTexture(devices.First().name, Screen.width, Screen.height);
        _webCam.Play();

        _canvas.texture = _webCam;
    }

    async void RecognitionLoop()
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();

        while (true)
        {
            try
            {
                var texture = ToTexture2D(_webCam);
                _recognizedObjects = (await _recognizer.DetectAsync(texture)).ToList();
            }
            catch (Exception)
            {
                Debug.Log($"Recognition loop interrupted...");
                break;
            }

            _updateTime = watch.ElapsedMilliseconds;
            watch.Restart();
        }
    }

    static Texture2D ToTexture2D(WebCamTexture cam)
    {
        Texture2D tex = new Texture2D(cam.width, cam.height);
        tex.SetPixels(cam.GetPixels());
        tex.filterMode = FilterMode.Trilinear;
        tex.Apply(true);
        return tex;
    }
}