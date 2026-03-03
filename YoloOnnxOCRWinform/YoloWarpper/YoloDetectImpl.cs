using Models;
using Sdcb.PaddleOCR;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using YoloOnnxOCRWinform.Models;
using YoloOnnxOCRWinform.YoloOnnx;

namespace YoloOnnxOCRWinform.YoloWarpper
{
    public class YoloDetectImpl : YoloDetectImplBase, IYoloModel, IYoloParallel
    {
        private YoloDetect yoloPredictor;

        public DetectResult DetectImage(string imgPath, PaddleOcrAll paddleOcrAll)
        {
            return DetectImage(imgPath, yoloPredictor, paddleOcrAll);
        }

        public void Dispose()
        {
            Dispose(yoloPredictor);
        }


        public ImagePreprocessModel[] GetPreLoadImages()
        {
            return yoloPredictor.GetPreLoadImages();
        }

        public void LoadModel(string modelPath, float confidence, float iou)
        {
            yoloPredictor = new YoloDetect(modelPath, confidence, iou);
        }



        public void PreLoadImages(BindingList<DataModel> list, Dictionary<string, string> dict)
        {
            yoloPredictor.PreLoadImages(list, dict);
        }

        public DetectResultModel Run(ImagePreprocessModel model, PaddleOcrAll paddleOcrAll)
        {
            return yoloPredictor.Run(model, paddleOcrAll);
        }

        public ShowResult SaveImage(DetectResultModel item)
        {
            return SaveImage(item, yoloPredictor);
        }

        public void EndPreload()
        {
            yoloPredictor.EndPreload();
        }


    }
}
