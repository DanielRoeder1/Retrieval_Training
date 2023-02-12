from datasets import load_dataset
from tqdm import tqdm
import csv


def write_section_text(entry, writer, thr_section_len = 100):
    """
    Extracts section text from paragraphs and writes to csv file
    :param paragraphs: list of paragraphs texts
    :param thr_section_len: threshold section length -> shortet section are removed
    """
    output = []
    buffer = ""
    for p in entry["text"]:
        if p.startswith("Section:::"):
            if len(buffer.split()) > thr_section_len:
                writer.writerow([entry["wikipedia_title"], buffer.replace("\n", "")])
            buffer = p
        else:
            buffer += " " +p
    return output


def process_dataset(data, file_name):
    """
    Process dataset and save to csv file
    :param data: dataset WikiDump
    :param file_name: csv file name
    """
    with open(file_name, "a", encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["title", "text"])
        
        for entry in tqdm(data):
            write_section_text(entry,writer)



if __name__ == "__main__":

    data = load_dataset("json", data_files=r'C:\Users\Daniel\Documents\RAG_thesis\data\kilt_knowledgesource.json', streaming=True)
    process_dataset(data["train"], "data/train2.csv")