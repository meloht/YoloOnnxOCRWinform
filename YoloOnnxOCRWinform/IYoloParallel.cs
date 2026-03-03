using Models;
using Sdcb.PaddleOCR;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using YoloOnnxOCRWinform.Models;
using YoloOnnxOCRWinform.YoloOnnx;

namespace YoloOnnxOCRWinform
{
    public interface IYoloParallel
    {
        void PreLoadImages(BindingList<DataModel> list, Dictionary<string, string> dict);

        ImagePreprocessModel[] GetPreLoadImages();

        DetectResultModel Run(ImagePreprocessModel model, PaddleOcrAll paddleOcrAll);

        void EndPreload();
    }
}
