import re

def clean_text(text):
    # ลบ Markdown link [text](url)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    # ลบ Markdown bold/italic **text** หรือ *text*
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    # ลบลิงก์ URL
    text = re.sub(r'http[s]?://\S+', '', text)
    # ลบรูปภาพ Markdown ![](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # ลบอีโมจิ/สัญลักษณ์พิเศษ (เบื้องต้น)
    text = re.sub(r'[\u2000-\u2BFF\U0001F300-\U0001FAD6]+', '', text)
    # ลบช่องว่างและบรรทัดซ้ำ
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    # ตัดช่องว่างหัว-ท้าย
    text = text.strip()
    return text

if __name__ == '__main__':
    input_path = 'c:/Users/sahac/coding/thesis/docs/index/chiang-mai.txt'
    output_path = 'c:/Users/sahac/coding/thesis/docs/index/chiang-mai.cleaned.txt'
    with open(input_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    cleaned = clean_text(raw)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    print(f'Cleaned file saved to {output_path}')
