namespace CommImageControl
{
    public class FormUtils
    {
        public static void Show(string fileName, string filePath)
        {
            FormShowImage showImage=new FormShowImage(fileName, filePath);
            showImage.Show();
        }

        public static void ShowOCR(string fileName, string ocr)
        {
            FormShowOCR showImage = new FormShowOCR(fileName, ocr);
            showImage.Show();
        }
    }
}
