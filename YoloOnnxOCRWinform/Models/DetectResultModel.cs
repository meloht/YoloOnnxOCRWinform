using System;
using System.Collections.Generic;
using System.Text;
using YoloOnnxOCRWinform.YoloOnnx;

namespace Models
{
    public class DetectResultModel
    {
        public List<Detection> DetectionList { get; set; }
        public string OCRResult { get; set; }
        public string FileName { get; set; }
        public string FilePath { get; set; }
        public bool IsOcr { get; set; }
    }
}
