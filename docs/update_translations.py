# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Script to fix broken Markdown links and front matter in language-specific directories zh, ko, ja, ru, de, fr, es, pt.

This script processes markdown files in language-specific directories (like /zh/). It finds Markdown links and checks
their existence. If a link is broken and does not exist in the language-specific directory but exists in the /en/
directory, the script updates the link to point to the corresponding file in the /en/ directory.

It also ensures that front matter keywords like 'comments:', 'description:', and 'keywords:' are not translated and
remain in English.
"""

import re
from pathlib import Path


class MarkdownLinkFixer:
    """Class to fix Markdown links and front matter in language-specific directories."""

    def __init__(self, base_dir, update_links=True, update_text=True):
        """Initialize the MarkdownLinkFixer with the base directory."""
        self.base_dir = Path(base_dir)
        self.update_links = update_links
        self.update_text = update_text
        self.md_link_regex = re.compile(r"\[([^]]+)]\(([^:)]+)\.md\)")

    @staticmethod
    def replace_front_matter(content, lang_dir):
        """Ensure front matter keywords remain in English."""
        english = ["comments", "description", "keywords"]
        translations = {
            "zh": ["评论", "描述", "关键词"],  # Mandarin Chinese (Simplified) warning, sometimes translates as 关键字
            "es": ["comentarios", "descripción", "palabras clave"],  # Spanish
            "ru": ["комментарии", "описание", "ключевые слова"],  # Russian
            "pt": ["comentários", "descrição", "palavras-chave"],  # Portuguese
            "fr": ["commentaires", "description", "mots-clés"],  # French
            "de": ["kommentare", "beschreibung", "schlüsselwörter"],  # German
            "ja": ["コメント", "説明", "キーワード"],  # Japanese
            "ko": ["댓글", "설명", "키워드"],  # Korean
            "hi": ["टिप्पणियाँ", "विवरण", "कीवर्ड"],  # Hindi
            "ar": ["التعليقات", "الوصف", "الكلمات الرئيسية"],  # Arabic
        }  # front matter translations for comments, description, keyword

        for term, eng_key in zip(translations.get(lang_dir.stem, []), english):
            content = (
                re.sub(rf"{term} *[：:].*", f"{eng_key}: true", content, flags=re.IGNORECASE)
                if eng_key == "comments"
                else re.sub(rf"{term} *[：:] *", f"{eng_key}: ", content, flags=re.IGNORECASE)
            )
        return content

    @staticmethod
    def replace_admonitions(content, lang_dir):
        """Ensure front matter keywords remain in English."""
        english = [
            "Note",
            "Summary",
            "Tip",
            "Info",
            "Success",
            "Question",
            "Warning",
            "Failure",
            "Danger",
            "Bug",
            "Example",
            "Quote",
            "Abstract",
            "Seealso",
            "Admonition",
        ]
        translations = {
            "en": english,
            "zh": [
                "笔记",
                "摘要",
                "提示",
                "信息",
                "成功",
                "问题",
                "警告",
                "失败",
                "危险",
                "故障",
                "示例",
                "引用",
                "摘要",
                "另见",
                "警告",
            ],
            "es": [
                "Nota",
                "Resumen",
                "Consejo",
                "Información",
                "Éxito",
                "Pregunta",
                "Advertencia",
                "Fracaso",
                "Peligro",
                "Error",
                "Ejemplo",
                "Cita",
                "Abstracto",
                "Véase También",
                "Amonestación",
            ],
            "ru": [
                "Заметка",
                "Сводка",
                "Совет",
                "Информация",
                "Успех",
                "Вопрос",
                "Предупреждение",
                "Неудача",
                "Опасность",
                "Ошибка",
                "Пример",
                "Цитата",
                "Абстракт",
                "См. Также",
                "Предостережение",
            ],
            "pt": [
                "Nota",
                "Resumo",
                "Dica",
                "Informação",
                "Sucesso",
                "Questão",
                "Aviso",
                "Falha",
                "Perigo",
                "Bug",
                "Exemplo",
                "Citação",
                "Abstrato",
                "Veja Também",
                "Advertência",
            ],
            "fr": [
                "Note",
                "Résumé",
                "Conseil",
                "Info",
                "Succès",
                "Question",
                "Avertissement",
                "Échec",
                "Danger",
                "Bug",
                "Exemple",
                "Citation",
                "Abstrait",
                "Voir Aussi",
                "Admonestation",
            ],
            "de": [
                "Hinweis",
                "Zusammenfassung",
                "Tipp",
                "Info",
                "Erfolg",
                "Frage",
                "Warnung",
                "Ausfall",
                "Gefahr",
                "Fehler",
                "Beispiel",
                "Zitat",
                "Abstrakt",
                "Siehe Auch",
                "Ermahnung",
            ],
            "ja": [
                "ノート",
                "要約",
                "ヒント",
                "情報",
                "成功",
                "質問",
                "警告",
                "失敗",
                "危険",
                "バグ",
                "例",
                "引用",
                "抄録",
                "参照",
                "訓告",
            ],
            "ko": [
                "노트",
                "요약",
                "팁",
                "정보",
                "성공",
                "질문",
                "경고",
                "실패",
                "위험",
                "버그",
                "예제",
                "인용",
                "추상",
                "참조",
                "경고",
            ],
            "hi": [
                "नोट",
                "सारांश",
                "सुझाव",
                "जानकारी",
                "सफलता",
                "प्रश्न",
                "चेतावनी",
                "विफलता",
                "खतरा",
                "बग",
                "उदाहरण",
                "उद्धरण",
                "सार",
                "देखें भी",
                "आगाही",
            ],
            "ar": [
                "ملاحظة",
                "ملخص",
                "نصيحة",
                "معلومات",
                "نجاح",
                "سؤال",
                "تحذير",
                "فشل",
                "خطر",
                "عطل",
                "مثال",
                "اقتباس",
                "ملخص",
                "انظر أيضاً",
                "تحذير",
            ],
        }

        for term, eng_key in zip(translations.get(lang_dir.stem, []), english):
            if lang_dir.stem != "en":
                content = re.sub(rf"!!! *{eng_key} *\n", f'!!! {eng_key} "{term}"\n', content, flags=re.IGNORECASE)
                content = re.sub(rf"!!! *{term} *\n", f'!!! {eng_key} "{term}"\n', content, flags=re.IGNORECASE)
            content = re.sub(rf"!!! *{term}", f"!!! {eng_key}", content, flags=re.IGNORECASE)
            content = re.sub(r'!!! *"', '!!! Example "', content, flags=re.IGNORECASE)

        return content

    @staticmethod
    def update_iframe(content):
        """Update the 'allow' attribute of iframe if it does not contain the specific English permissions."""
        english = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        pattern = re.compile(f'allow="(?!{re.escape(english)}).+?"')
        return pattern.sub(f'allow="{english}"', content)

    def link_replacer(self, match, parent_dir, lang_dir, use_abs_link=False):
        """Replace broken links with corresponding links in the /en/ directory."""
        text, path = match.groups()
        linked_path = (parent_dir / path).resolve().with_suffix(".md")

        if not linked_path.exists():
            en_linked_path = Path(str(linked_path).replace(str(lang_dir), str(lang_dir.parent / "en")))
            if en_linked_path.exists():
                if use_abs_link:
                    # Use absolute links WARNING: BUGS, DO NOT USE
                    docs_root_relative_path = en_linked_path.relative_to(lang_dir.parent)
                    updated_path = str(docs_root_relative_path).replace("en/", "/../")
                else:
                    # Use relative links
                    steps_up = len(parent_dir.relative_to(self.base_dir).parts)
                    updated_path = Path("../" * steps_up) / en_linked_path.relative_to(self.base_dir)
                    updated_path = str(updated_path).replace("/en/", "/")

                print(f"Redirecting link '[{text}]({path})' from {parent_dir} to {updated_path}")
                return f"[{text}]({updated_path})"
            else:
                print(f"Warning: Broken link '[{text}]({path})' found in {parent_dir} does not exist in /docs/en/.")

        return match.group(0)

    @staticmethod
    def update_html_tags(content):
        """Updates HTML tags in docs."""
        alt_tag = "MISSING"

        # Remove closing slashes from self-closing HTML tags
        pattern = re.compile(r"<([^>]+?)\s*/>")
        content = re.sub(pattern, r"<\1>", content)

        # Find all images without alt tags and add placeholder alt text
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
        content, num_replacements = re.subn(
            pattern, lambda match: f"![{match.group(1) or alt_tag}]({match.group(2)})", content
        )

        # Add missing alt tags to HTML images
        pattern = re.compile(r'<img\s+(?!.*?\balt\b)[^>]*src=["\'](.*?)["\'][^>]*>')
        content, num_replacements = re.subn(
            pattern, lambda match: match.group(0).replace(">", f' alt="{alt_tag}">', 1), content
        )

        return content

    def process_markdown_file(self, md_file_path, lang_dir):
        """Process each markdown file in the language directory."""
        print(f"Processing file: {md_file_path}")
        with open(md_file_path, encoding="utf-8") as file:
            content = file.read()

        if self.update_links:
            content = self.md_link_regex.sub(lambda m: self.link_replacer(m, md_file_path.parent, lang_dir), content)

        if self.update_text:
            content = self.replace_front_matter(content, lang_dir)
            content = self.replace_admonitions(content, lang_dir)
            content = self.update_iframe(content)
            content = self.update_html_tags(content)

        with open(md_file_path, "w", encoding="utf-8") as file:
            file.write(content)

    def process_language_directory(self, lang_dir):
        """Process each language-specific directory."""
        print(f"Processing language directory: {lang_dir}")
        for md_file in lang_dir.rglob("*.md"):
            self.process_markdown_file(md_file, lang_dir)

    def run(self):
        """Run the link fixing and front matter updating process for each language-specific directory."""
        for subdir in self.base_dir.iterdir():
            if subdir.is_dir() and re.match(r"^\w\w$", subdir.name):
                self.process_language_directory(subdir)


if __name__ == "__main__":
    # Set the path to your MkDocs 'docs' directory here
    docs_dir = str(Path(__file__).parent.resolve())
    fixer = MarkdownLinkFixer(docs_dir, update_links=True, update_text=True)
    fixer.run()
