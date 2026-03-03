using Models;
using Sdcb.PaddleOCR;
using System;
using System.Collections.Generic;
using System.Text;
using YoloOnnxOCRWinform.Models;

namespace YoloOnnxOCRWinform
{
    public interface IYoloModel : IDisposable
    {
        void LoadModel(string modelPath, float confidence, float iou);
        ShowResult SaveImage(DetectResultModel item);
        DetectResult DetectImage(string imgPath, PaddleOcrAll paddleOcrAll);

    }
}
