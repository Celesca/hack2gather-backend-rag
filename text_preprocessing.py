import re
from fixthaipdf import clean

class TextPreprocessor:
    @staticmethod
    def clean_text_refined(text):
        # 1. Remove extra spaces between Thai characters
        text = re.sub(r'([ก-๙])\s+([ก-๙])', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\s+([.,!?()])', r'\1', text)
        text = re.sub(r'([.,!?()])\s+', r'\1 ', text)
        text = re.sub(r'\s+', '', text)
        return text

    @staticmethod
    def fix_thai_text(text):
        pattern = r'([ก-ฮ])([ิ-ู]|[เ-แ]|[ั]|[็]|[์]|[่-๋])'
        fixed_text = re.sub(pattern, r'\1\2', text)
        pattern2 = r'([ิ-ู])([่-๋])'
        fixed_text = re.sub(pattern2, r'\1\2', fixed_text)
        return fixed_text

    @staticmethod
    def remove_newline_char(text):
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        text = text.replace('\x0c', ' ')
        return text

    @staticmethod
    def preprocess_text(text):
        # text = TextPreprocessor.remove_newline_char(text)
        text = TextPreprocessor.clean_text_refined(text)

        return text
    
    def process_document(self, document):
        document.page_content = self.preprocess_text(document.page_content)
        return document