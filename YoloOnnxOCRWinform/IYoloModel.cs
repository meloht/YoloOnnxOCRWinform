using System;
using System.Collections.Generic;
using System.Text;
using YoloOnnxOCRWinform.Models;

namespace YoloOnnxOCRWinform
{
    public interface IYoloModel : IDisposable
    {
        void LoadModel(string modelPath, float confidence, float iou);
        string SaveImage(FileRowItem item);
        string DetectImage(string imgPath);
    }
}
