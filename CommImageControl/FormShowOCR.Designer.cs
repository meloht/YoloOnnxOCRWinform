namespace CommImageControl
{
    partial class FormShowOCR
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            textBoxOcr = new TextBox();
            SuspendLayout();
            // 
            // textBoxOcr
            // 
            textBoxOcr.Dock = DockStyle.Fill;
            textBoxOcr.Location = new Point(0, 0);
            textBoxOcr.Multiline = true;
            textBoxOcr.Name = "textBoxOcr";
            textBoxOcr.ScrollBars = ScrollBars.Vertical;
            textBoxOcr.Size = new Size(800, 450);
            textBoxOcr.TabIndex = 0;
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(7F, 17F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(800, 450);
            Controls.Add(textBoxOcr);
            Name = "Form1";
            Text = "Show OCR Result";
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private TextBox textBoxOcr;
    }
}