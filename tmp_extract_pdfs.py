from pathlib import Path
from pypdf import PdfReader

files = [
    Path(r"C:\Users\Lenovo\Downloads\Критерии_оценки_—_Мониторинг_природных_пожаров_КосмоХакатон_2026 (1).pdf"),
    Path(r"C:\Users\Lenovo\Downloads\Постановка_кейса_—_Мониторинг_природных_пожаров_КосмоХакатон_2026.pdf"),
]
for path in files:
    reader = PdfReader(path)
    print(f"\n### {path.name} pages={len(reader.pages)}")
    for index, page in enumerate(reader.pages, 1):
        print(f"\n--- PAGE {index} ---\n{page.extract_text() or ''}")
