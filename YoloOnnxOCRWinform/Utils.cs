using OpenCvSharp;
using Sdcb.PaddleInference;
using Sdcb.PaddleOCR;
using Sdcb.PaddleOCR.Models;
using Sdcb.PaddleOCR.Models.Local;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using YoloOnnxOCRWinform.Models;
using YoloOnnxOCRWinform.YoloOnnx;

namespace YoloOnnxOCRWinform
{
    public class Utils
    {
        public const string TextClassName = "text";
        public static DetectResult GetResult(List<Detection> list, Mat inputImage)
        {
            if (list == null || list.Count == 0)
                return new DetectResult(string.Empty, string.Empty);

            var dict = list.GroupBy(p => p.ClassName).Select(p => $"{p.Count()} {p.Key}").ToList();
            string confs = string.Join(", ", list.Select(p => Math.Round(p.Confidence, 2)));
            string result = $"{string.Join(", ", dict)} [{confs}]";

            var OCRResult = GetOCRResult(list, inputImage);

            return new DetectResult(result, OCRResult.ocrResult);
        }

        public static OCRResult GetOCRResult(List<Detection> list, Mat inputImage)
        {
            StringBuilder sb = new StringBuilder();

            FullOcrModel model = LocalFullModels.EnglishV4;

            using PaddleOcrAll all = new PaddleOcrAll(model, PaddleDevice.Mkldnn())
            {
                AllowRotateDetection = false, /* 允许识别有角度的文字 */
                Enable180Classification = false, /* 允许识别旋转角度大于90度的文字 */
            };
            bool isOCR = false;

            foreach (var detection in list)
            {
                if (detection.ClassName != TextClassName)
                {
                    continue;
                }
                isOCR = true;
                // 4. 执行裁剪（SubMat：提取指定ROI区域，不会复制数据，效率高）
                using (Mat croppedImage = inputImage.SubMat(detection.Box))
                {
                    PaddleOcrResult result = all.Run(croppedImage);

                    foreach (PaddleOcrResultRegion region in result.Regions)
                    {
                        sb.Append(region.Text).Append(" ");
                    }
                    sb.AppendLine();
                }
            }
            return new OCRResult(sb.ToString(), isOCR);
        }


    }
}
