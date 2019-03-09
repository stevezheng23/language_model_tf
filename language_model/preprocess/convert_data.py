import argparse
import os
import os.path
import re
import json
import nltk

def add_arguments(parser):
    parser.add_argument("--dataset", help="dataset", required=True)
    parser.add_argument("--input_dir", help="input directory", required=True)
    parser.add_argument("--output_dir", help="output directory", required=True)
    parser.add_argument("--min_seq_len", help="mininum sequence length", required=False, type=int, default=10)
    parser.add_argument("--max_seq_len", help="maximum sequence length", required=False, type=int, default=1000)

def normalize_text(text, lower_case=True, remove_punc=False):
    def process_token(tokens):
        special = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        pattern = "([{}])".format("".join(special))
        processed_tokens = []
        for token in tokens:
            token = token.replace("''", '"').replace("``", '"')
            processed_tokens.extend(re.split(pattern, token))
        
        return processed_tokens
    
    def remove_punctuation(tokens):
        exclude = set(string.punctuation)
        return [token for token in tokens if token not in exclude]
    
    def fix_white_space(tokens):
        return [token for token in tokens if token and not token.isspace()]
    
    sents = nltk.sent_tokenize(text)
    norm_sents = []
    for sent in sents:
        words = nltk.word_tokenize(sent)
        words = process_token(words)
        if remove_punc:
            words = remove_punctuation(words)
        
        words = fix_white_space(words)
        norm_sents.append(' '.join(words))
    
    norm_text = ' '.join(norm_sents)
    if lower_case:
        norm_text = norm_text.lower()
    
    return norm_text.strip()

def preprocess_wikipedia(input_dir,
                         output_dir,
                         min_seq_len,
                         max_seq_len):
    if not os.path.exists(input_dir):
        raise FileNotFoundError("input dir not found")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for file_name in os.listdir(input_dir):
        input_file = os.path.join(input_dir, file_name)
        if not os.path.exists(input_file) or not os.path.isfile(input_file):
            continue
        
        print("process file: {0}".format(file_name))
        
        processed_lines = []
        with open(input_file, "rb") as file:
            for line in file:
                input_data = json.loads(line.decode("utf-8").strip())
                norm_text = normalize_text(input_data["text"], False, False)
                norm_tokens = norm_text.split(" ")

                if len(norm_tokens) < min_seq_len:
                    continue

                while len(norm_tokens) > 0:
                    processed_lines.append(" ".join(norm_tokens[:max_seq_len]))
                    norm_tokens = norm_tokens[max_seq_len:]
        
        output_file = os.path.join(output_dir, "{0}.{1}".format(os.path.splitext(file_name)[0], "processed"))
        with open(output_file, "wb") as file:
            for processed_line in processed_lines:
                file.write("{0}\r\n".format(processed_line).encode("utf-8"))

def preprocess_bookcorpus(input_dir,
                          output_dir,
                          min_seq_len,
                          max_seq_len):
    if not os.path.exists(input_dir):
        raise FileNotFoundError("input dir not found")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for file_name in os.listdir(input_dir):
        input_file = os.path.join(input_dir, file_name)
        if not os.path.exists(input_file) or not os.path.isfile(input_file):
            continue
        
        print("process file: {0}".format(file_name))
        
        processed_lines = []
        with open(input_file, "rb") as file:
            raw_text = file.read().decode("utf-8")
            norm_text = normalize_text(raw_text, False, False)
            norm_tokens = norm_text.split(" ")
            
            if len(norm_tokens) < min_seq_len:
                continue
            
            while len(norm_tokens) > 0:
                processed_lines.append(" ".join(norm_tokens[:max_seq_len]))
                norm_tokens = norm_tokens[max_seq_len:]
        
        output_file = os.path.join(output_dir, "{0}.{1}".format(os.path.splitext(file_name)[0], "processed"))
        with open(output_file, "wb") as file:
            for processed_line in processed_lines:
                file.write("{0}\r\n".format(processed_line).encode("utf-8"))

def main(args):
    if args.dataset == "wikipedia":
        preprocess_wikipedia(args.input_dir, args.output_dir, args.min_seq_len, args.max_seq_len)
    elif args.dataset == "bookcorpus":
        preprocess_bookcorpus(args.input_dir, args.output_dir, args.min_seq_len, args.max_seq_len)
    else:
        raise ValueError("pre-processing on dataset: {0} is not supported".format(data_source))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
