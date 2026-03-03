using Models;
using Sdcb.PaddleOCR;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using YoloOnnxOCRWinform.Models;
using YoloOnnxOCRWinform.YoloOnnx;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace YoloOnnxOCRWinform
{
    public class ViewPresenter
    {
        IFormProgress _formProgress;

        protected BindingList<DataModel> _bindingSource = new BindingList<DataModel>();
        protected Dictionary<string, string> _dictFile = [];
        private Dictionary<string, DetectResultModel> _dictResult = new Dictionary<string, DetectResultModel>();
        private System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();


        public ViewPresenter(IFormProgress progress)
        {
            _formProgress = progress;
        }

        public void InitDataGridColumn(List<string> files, Dictionary<string, string> dict)
        {
            _dictFile.Clear();
            _dictFile = dict;
            _bindingSource.Clear();
            _dictResult.Clear();
            SetDataGridColumns();
            foreach (var fileName in files)
            {
                DataModel model = new DataModel();
                model.FileName = fileName;
                _bindingSource.Add(model);
            }
            if (_bindingSource.Count == 0)
            {
                MessageBox.Show("There are no pictures in this directory!", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            _formProgress.DataGridList.DataSource = _bindingSource;


        }

        protected void AddColumn(string colName, int width, DataGridView dataGridView)
        {
            DataGridViewTextBoxColumn column = new DataGridViewTextBoxColumn();
            column.HeaderText = colName;
            column.Name = $"Column{colName}";
            column.ReadOnly = true;
            column.Width = width;
            column.DataPropertyName = colName;

            dataGridView.Columns.Add(column);
        }
        protected void SetDataGridColumns()
        {
            _formProgress.DataGridList.Columns.Clear();
            AddColumn("FileName", 350, _formProgress.DataGridList);
            AddColumn("DetectionResult", 240, _formProgress.DataGridList);
            AddColumn("DetectionText", 300, _formProgress.DataGridList);
            AddColumn("ExecuteTime", 200, _formProgress.DataGridList);
            AddColumn("ErrorLog", 400, _formProgress.DataGridList);

        }

        public void Process(IYoloModel yoloPredictor, ExcuteType excuteType, PaddleOcrAll paddleOcrAll)
        {
            if (yoloPredictor is IYoloParallel && ExcuteType.Parallel == excuteType)
            {

                ProcessParallel(yoloPredictor, paddleOcrAll);
            }
            else
            {
                ProcessSequence(yoloPredictor, paddleOcrAll);
            }

        }

        private void ProcessSequence(IYoloModel yoloPredictor, PaddleOcrAll paddleOcrAll)
        {
            int idx = 0;
            int total = _bindingSource.Count;


            foreach (var item in _bindingSource)
            {

                try
                {
                    string filePath = _dictFile[item.FileName];
                    GetDetectResult(yoloPredictor, item, filePath, paddleOcrAll);
                    _formProgress.ShowProgress(idx * 100 / total, $"{idx}/{total}");

                }
                catch (Exception ex)
                {
                    ShowError(item, ex.Message);
                }

                idx++;
            }

            _formProgress.ShowProgress(idx * 100 / total, $"{idx}/{total}");
        }

        private void ProcessParallel(IYoloModel yoloPredictor, PaddleOcrAll paddleOcrAll)
        {
            IYoloParallel yolo = (IYoloParallel)yoloPredictor;
            yolo.PreLoadImages(_bindingSource, _dictFile);
            int idx = 0;
            int total = _bindingSource.Count;

            for (; idx < total;)
            {
                var list = yolo.GetPreLoadImages();
                foreach (var item1 in list)
                {
                    try
                    {
                        _stopwatch.Start();
                        DetectResultModel detectResult = yolo.Run(item1, paddleOcrAll);

                        _stopwatch.Stop();
                        item1.model.ExecuteTime = $"{_stopwatch.Elapsed.TotalMilliseconds}ms";
                        _stopwatch.Reset();

                        UpdateDetectResult(detectResult);

                    }
                    catch (Exception ex)
                    {
                        ShowError(item1.model, ex.Message);
                    }
                    idx++;
                    _formProgress.ShowProgress(idx * 100 / total, $"{idx}/{total}");
                }
                Thread.Sleep(50);
            }

        }



        private void GetDetectResult(IYoloModel yoloPredictor, DataModel model, string filePath, PaddleOcrAll paddleOcrAll)
        {
            _stopwatch.Start();
            var result = yoloPredictor.DetectImage(filePath, paddleOcrAll);
            model.DetectionResult = result.result;
            model.DetectionText = result.ocr;
            _stopwatch.Stop();
            model.ExecuteTime = $"{_stopwatch.Elapsed.TotalMilliseconds}ms";
            _stopwatch.Reset();

            UpdateDetectResult(result.ocr, model.FileName, result.list, filePath);

        }

        private void UpdateDetectResult(string ocr, string fileName, List<Detection> list, string filePath)
        {
            if (_dictResult.ContainsKey(filePath))
            {
                var data = _dictResult[filePath];
                if (data != null)
                {
                    data.OCRResult = ocr;
                    data.DetectionList = list;
                    data.FilePath = filePath;
                    data.FileName = fileName;
                }
            }
            else
            {
                DetectResultModel data = new DetectResultModel();
                data.DetectionList = list;
                data.OCRResult = ocr;
                data.FilePath = filePath;
                data.FileName = fileName;
                _dictResult.Add(filePath, data);
            }
        }

        private void UpdateDetectResult(DetectResultModel detectResult)
        {
            if (_dictResult.ContainsKey(detectResult.FilePath))
            {
                var data = _dictResult[detectResult.FilePath];
                if (data != null)
                {
                    _dictResult[detectResult.FilePath] = detectResult;
                }
            }
            else
            {
                _dictResult.Add(detectResult.FilePath, detectResult);
            }
        }

        public DetectResultModel GetSelectRowData(DataGridViewRow row)
        {
            var item = row.DataBoundItem as DataModel;
            if (item != null)
            {
                string path = _dictFile[item.FileName];
                if (_dictResult.ContainsKey(path))
                {
                    return _dictResult[path];

                }
            }
            return null;
        }


        protected void ShowError(DataModel item, string error)
        {
            _formProgress?.DataGridList?.Invoke(new Action(() =>
            {
                if (!string.IsNullOrEmpty(item.ErrorLog))
                {
                    item.ErrorLog = $"{item.ErrorLog} {error}";
                }
                else
                {
                    item.ErrorLog = error;
                }

            }));
        }

        protected void UpdateImageItemModel(DataModel item, DataModel model)
        {
            item.ErrorLog = model.ErrorLog;
            item.DetectionResult = model.DetectionResult;
            item.ExecuteTime = model.ExecuteTime;

        }


    }
}
