using Models;
using OpenCvSharp;
using Sdcb.PaddleOCR;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using YoloOnnxOCRWinform.Models;

namespace YoloOnnxOCRWinform.YoloOnnx
{
    public interface IYoloDetect : IDisposable
    {
        void DrawDetections(Mat inputImage, List<Detection> list);
        List<Detection> Run(Mat inputImage);
        DetectResultModel Run(ImagePreprocessModel model, PaddleOcrAll paddleOcrAll);
        void PreLoadImages(BindingList<DataModel> list, Dictionary<string, string> dict);

        ImagePreprocessModel[] GetPreLoadImages();

        void EndPreload();


        
    }
}
