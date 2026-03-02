using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace YoloOnnxOCRWinform.Models
{
    public class DataModel : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }


        private string _fileName;
        private string _detectionResult;
        private string _detectionText;
        private string _executeTime;
        private string _errorLog;

        public string FileName
        {
            get => _fileName;
            set
            {
                _fileName = value;
                OnPropertyChanged(nameof(FileName));
            }
        }

        public string DetectionResult
        {
            get => _detectionResult;
            set
            {
                _detectionResult = value;
                OnPropertyChanged(nameof(DetectionResult));
            }
        }

        public string DetectionText
        {
            get => _detectionText;
            set
            {
                _detectionText = value;
                OnPropertyChanged(nameof(DetectionText));
            }
        }

        public string ExecuteTime
        {
            get => _executeTime;
            set
            {
                _executeTime = value;
                OnPropertyChanged(nameof(ExecuteTime));
            }
        }
        public string ErrorLog
        {
            get => _errorLog;
            set
            {
                _errorLog = value;
                OnPropertyChanged(nameof(ErrorLog));
            }
        }
    }
}
