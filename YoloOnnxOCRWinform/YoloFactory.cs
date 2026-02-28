using System;
using System.Collections.Generic;
using System.Text;
using YoloOnnxOCRWinform.YoloWarpper;

namespace YoloOnnxOCRWinform
{
    public enum YoloWarpperType
    {
        YoloDetect,
        YoloDetectOrt
    }
    public static class YoloFactory
    {

        public static IYoloModel Create(YoloWarpperType yoloWarpperType)
        {
            switch (yoloWarpperType)
            {
                case YoloWarpperType.YoloDetect:
                    return new YoloDetectImpl();
                case YoloWarpperType.YoloDetectOrt:
                    return new YoloDetectOrtValImpl();
                default:
                    return new YoloDetectOrtValImpl();
            }
        }
    }
}
