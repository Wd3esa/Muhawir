"""Fixed user-facing texts. Answers themselves only come from passages."""

LANGS = ("ar", "en")
STYLES = ("kids", "youth", "extended", "newcomer")

TEXT = {
    "ar": {
        "abstain": "لم أجد في المصادر المعتمدة المتاحة لي ما يجيب عن هذا السؤال. "
                   "يمكنك سؤال مختص في العلم الشرعي.",
        "personal_case": "سؤالك عن حالة شخصية، والحكم فيها يحتاج فتوى من مختص يسمع تفاصيلها. "
                         "أنصحك بسؤال جهة فتوى مؤهلة في بلدك.",
        "personal_case_info": "هذه معلومات عامة من المصادر، وليست حكمًا في حالتك:",
        "judging_people": "لا أحكم على أشخاص أو جماعات بعينهم، فهذا خارج ما أقدّمه. "
                          "يمكنني أن أعرض لك ما تقوله المصادر المعتمدة عن المفاهيم نفسها.",
        "override_attempt": "لا أستطيع تغيير طريقتي: أجيب من المصادر المعتمدة فقط ولا أفتي برأيي. "
                            "إن كان لديك سؤال، فاكتبه وسأبحث عنه في المصادر، أو اسأل مختصًا.",
        "translation_pending": "",
        "synthetic": "بيانات تجريبية مصطنعة للاختبار، وليست نصوصًا دينية.",
        "too_long": "السؤال طويل جدًا. اختصره من فضلك.",
        "empty": "اكتب سؤالك أولًا.",
    },
    "en": {
        "abstain": "I could not find anything in the approved sources available to me that answers "
                   "this question. "
                   "You may ask a qualified scholar.",
        "personal_case": "Your question is about a personal situation. A ruling on it needs a fatwa "
                         "from a qualified scholar who hears the details. Please ask a qualified "
                         "fatwa body in your country.",
        "personal_case_info": "This is general information from the sources, not a ruling on your case:",
        "judging_people": "I do not pass judgement on specific people or groups; that is outside what "
                          "I offer. I can show what the approved sources say about the concepts themselves.",
        "override_attempt": "I cannot change how I work: I answer only from approved sources and do not "
                            "give fatwas of my own. Ask your question and I will look for it in the "
                            "sources, or ask a qualified scholar.",
        "translation_pending": "The quotation is shown in its original language. Translation will be "
                               "added once a language model is connected.",
        "synthetic": "Synthetic test data, not religious texts.",
        "too_long": "The question is too long. Please shorten it.",
        "empty": "Please type your question first.",
    },
}
