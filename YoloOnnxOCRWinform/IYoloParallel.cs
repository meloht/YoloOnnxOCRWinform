using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using YoloOnnxOCRWinform.Models;

namespace YoloOnnxOCRWinform
{
    public interface IYoloParallel
    {
        void PreLoadImages(BindingList<DataModel> list, Dictionary<string, string> dict);

        ImagePreprocessModel[] GetPreLoadImages();

        void Run(ImagePreprocessModel model);

        void EndPreload();
    }
}
