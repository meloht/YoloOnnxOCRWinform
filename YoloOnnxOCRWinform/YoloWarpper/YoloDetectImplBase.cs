using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloOnnxOCRWinform.Models;
using YoloOnnxOCRWinform.YoloOnnx;

namespace YoloOnnxOCRWinform.YoloWarpper
{
    public class YoloDetectImplBase
    {
        protected string DetectImage(string imgPath, IYoloDetect yoloPredictor)
        {
            using Mat inputImage = Cv2.ImRead(imgPath);
            var data = yoloPredictor.Run(inputImage);
            return Utils.GetResult(data);
        }

        protected void Dispose(IYoloDetect yoloPredictor)
        {
            yoloPredictor?.Dispose();
        }
        protected ShowResult SaveImage(FileRowItem item, IYoloDetect yoloPredictor)
        {
            using Mat inputImage = Cv2.ImRead(item.FilePath);

            var result = yoloPredictor.Run(inputImage);
            var ocrResult = Utils.GetOCRResult(result, inputImage);
            yoloPredictor.DrawDetections(inputImage, result);
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
            return new ShowResult(path, ocrResult.Item2, ocrResult.Item1);
        }
    }
}
