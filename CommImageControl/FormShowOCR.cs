using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace CommImageControl
{
    public partial class FormShowOCR : Form
    {
        public FormShowOCR(string fileName,string str)
        {
            InitializeComponent();
            this.Text = fileName;
            this.textBoxOcr.Text = str;
        }
    }
}
