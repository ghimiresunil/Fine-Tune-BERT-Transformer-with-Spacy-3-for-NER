import re

import fitz

from clean_text import clean_text


def get_text_percentage(file_name: str) -> float:
    total_page_area = 0.0
    total_text_area = 0.0

    doc = fitz.open(file_name)
    for page_num, page in enumerate(doc):
        total_page_area = total_page_area + abs(page.rect)
        text_area = 0.0
        for b in page.get_text_blocks():
            r = fitz.Rect(b[:4])  # rectangle where block text appears
            text_area = text_area + abs(r)
        total_text_area = total_text_area + text_area
        doc.close()
        # print(total_text_area / total_page_area)
        return total_text_area / total_page_area


def pdf_to_text(file_path, dolower):
    """
    Takes filepath and extracts
    the plain text from pdf for
    training the word to vec model
    :param file_path :type str
    :return:text   :type str
    """
    doc = fitz.open(file_path)
    number_of_pages = doc.page_count
    text = ""
    links = ""
    for i in range(0, number_of_pages):
        if (
            get_text_percentage(file_path) > 0.44
            and get_text_percentage(file_path) < 0.50
        ):
            page = doc.load_page(i)
            pagetext = page.get_text("text", sort=True, flags=16)
            pagelinks = [link["uri"] for link in page.links() if "uri" in link]
            text += pagetext
            links += links

        else:
            page = doc.load_page(i)
            pagetext = page.get_text("text", sort=False, flags=16)
            pagelinks = " ".join(
                [link["uri"] for link in page.links() if "uri" in link]
            )
            text += pagetext
            links += pagelinks

    text = clean_text(text, dolower)
    extract_link = clean_text(links, dolower)
    link_regex = re.compile(
        r"((github.com/[^ |^\n]+)|(github:[^ |^\n]+)|(github/[^ |^\n]+)|(linkedin.com/[^ |^\n]+)|(linkedin:[^ |^\n]+)|(linkedin/[^ |^\n]+))"
    )
    links = " ".join(
        {str(match.group()) for match in re.finditer(link_regex, extract_link)}
    )
    text = "".join(text + "\nextracted links\n" + links)
    return text


if __name__ == "__main__":
    path = "testresume/Ali, Mohammad_Taha - 2022-06-23 07-36-29.pdf"
    text = pdf_to_text(path, True)
    print(text)