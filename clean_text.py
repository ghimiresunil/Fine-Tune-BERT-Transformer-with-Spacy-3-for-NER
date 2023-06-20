import re


def clean_text(text, dolower):
    """
    Accepts the plain text and makes
    use of regex for cleaning the noise
    :param: text :type:str
    :return:cleaned text :type str
    """
    if dolower == True:
        text = text.lower()
    text = re.sub(" +", " ", text)
    text = re.sub("\n+", "\n", text)
    text = re.sub("\t+", "\t", text)
    text = re.sub(r"\uf0b7", " ", text)
    text = re.sub(r"\(cid:\d{0,3}\)", " ", text)
    text = [i.strip() for i in text.splitlines()]
    text = "\n".join(text)
    text = re.sub("\n+", "\n", text)
    text = re.sub(r"• ", " ", text)
    text = text.encode("ascii", errors="ignore").decode("utf-8")
    text = text.replace("// ", "")
    return text