using System;
using System.Collections.Generic;
using System.Text;

namespace YoloOnnxOCRWinform
{
    public interface IFormProgress
    {
        void ShowProgress(int val, string info);


        DataGridView DataGridList { get; }
    }
}
