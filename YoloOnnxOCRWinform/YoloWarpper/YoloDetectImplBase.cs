using Models;
using OpenCvSharp;
using Sdcb.PaddleOCR;
using System;
using System.Collections.Generic;
using System.Text;
using YoloOnnxOCRWinform.Models;
using YoloOnnxOCRWinform.YoloOnnx;

namespace YoloOnnxOCRWinform.YoloWarpper
{
    public class YoloDetectImplBase
    {
        protected DetectResult DetectImage(string imgPath, IYoloDetect yoloPredictor, PaddleOcrAll paddleOcrAll)
        {
            using Mat inputImage = Cv2.ImRead(imgPath);
            var data = yoloPredictor.Run(inputImage);

            return Utils.GetResult(data, inputImage, paddleOcrAll);
        }

        protected void Dispose(IYoloDetect yoloPredictor)
        {
            yoloPredictor?.Dispose();
        }
        protected ShowResult SaveImage(DetectResultModel item, IYoloDetect yoloPredictor)
        {
            using Mat inputImage = Cv2.ImRead(item.FilePath);

            yoloPredictor.DrawDetections(inputImage, item.DetectionList);
            string folder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "temp");
            if (!Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);
            }
            string path = Path.Combine(folder, item.FileName);
            if (File.Exists(path))
            {
                File.Delete(path);
            }
            Cv2.ImWrite(path, inputImage);
            return new ShowResult(path, item.OCRResult, item.IsOcr);
        }
    }
}
