import os
import pandas as pd
import time
import json
import fitz  # PyMuPDF
import pandas as pd
# from pdfminer.high_level import extract_text
import os
import pickle 
import shutil
import os
from itertools import combinations
import xml.etree.ElementTree as ET
from datetime import datetime
from rdflib import Graph
import rdflib
from rdflib import Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS


import json
import re
from statistics import mean
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize


# from pdf2image import convert_from_path # type: ignore
# import cv2
# import layoutparser as lp # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import io
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import requests
import re
# import pytesseract
import io
from PIL import Image
import glob
from bs4 import BeautifulSoup, NavigableString
from unidecode import unidecode




#DOCS: https://platform.openai.com/docs/guides/vision
import openai
# from openai import OpenAI

nltk.download('punkt')
nltk.download('stopwords')


def chunk_text_by_sentences(text, sentences_per_chunk=5):
    """
    Splits the text into chunks, each containing a fixed number of sentences.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) == sentences_per_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks



def create_owl_file(title, glossary, taxonomy, relationships):
    # Create a new graph
    g = rdflib.Graph()

    # Define namespace
    namespace = Namespace("http://example.org/ontology/")

    # Title as a Literal
    ontology_title = URIRef(namespace["OntologyTitle"])
    g.add((ontology_title, RDFS.label, Literal(title.strip())))

    # Add glossary terms
    for entry in glossary.strip().split("\n\n"):
        if entry:
            term, definition = entry.split(":", 1)
            term_uri = URIRef(namespace[term.strip().replace(" ", "_")])
            g.add((term_uri, RDFS.label, Literal(term.strip())))
            g.add((term_uri, RDFS.comment, Literal(definition.strip())))

    # Add taxonomy
    def add_taxonomy(parent_uri, structure, parent_indent=0):
        nonlocal g, namespace
        lines = structure.strip().split("\n")
        for i, line in enumerate(lines):
            if line.strip():
                indent_level = len(line) - len(line.lstrip())
                term = line.strip()
                term_uri = URIRef(namespace[term.replace(" ", "_")])
                if indent_level == parent_indent:
                    g.add((term_uri, RDFS.subClassOf, parent_uri))
                elif indent_level > parent_indent:
                    # Recursively add more specific sub-classes
                    sub_structure = "\n".join(lines[i:])
                    add_taxonomy(term_uri, sub_structure, indent_level)
                    break  # Exit the loop after handling the nested structure

    # Initialize taxonomy building
    add_taxonomy(ontology_title, taxonomy)

    # Add relationship mappings
    for line in relationships.strip().split("\n\n"):
        if line:
            description = line.strip()
            # Example parse: assumes "A is related to B" from description
            # This needs to be customized based on actual relationship format
            entities = description.split('"')[1::2]  # Getting terms between quotes
            for i in range(len(entities)-1):
                entity1_uri = URIRef(namespace[entities[i].replace(" ", "_")])
                entity2_uri = URIRef(namespace[entities[i+1].replace(" ", "_")])
                relation_uri = URIRef(namespace["is_related_to"])
                g.add((entity1_uri, relation_uri, entity2_uri))

    # Serialize the graph in RDF/XML format
    owl_data = g.serialize(format='xml')
    print(owl_data)
    
    print("ENDDDD!!!")
    return owl_data

def generate_ontology(text, title):
    # openai.api_key = 'your-api-key'  # Replace 'your-api-key' with your actual OpenAI API key
    print(text)
    print(title)
    
    stop_words = set(stopwords.words('english'))
    #TODO
    
    
    def find_similar_words(lists):
        similar_words = []
        for lst in lists:
            for word in lst:
                if word not in similar_words:
                    is_similar = False
                    for similar_lst in similar_words:
                        if word.lower() in [s.lower() for s in similar_lst]:
                            similar_lst.append(word)
                            is_similar = True
                            break
                    if not is_similar:
                        similar_words.append([word])
        return similar_words

    
    def remove_stop_words(text):
        # Tokenize the text into words
        words = word_tokenize(text)
        
        # Filter out the stop words
        filtered_words = [word for word in words if word.lower() not in stop_words]
        # print(filtered_words)
        # input()
        # Return the filtered list of words
        return filtered_words
    
    def clean_and_format(words):
        # Remove unnecessary characters and formatting issues
        cleaned_words = [word for word in words if word not in ["{", "}", "``", "''", "'", ",", ":"]]
        
        # Join words into a single string
        text = " ".join(cleaned_words)
        
        # Replace unwanted punctuations or artifacts
        text = text.replace(" :", ":").replace(" .", ".").replace(" ,", ",").replace(" '", "'").replace("$ ", "$")
        
        return text
    

    def get_glossary(text):
        response = openai.chat.completions.create(
            model="gpt-4",  # or "gpt-4" if available,
            temperature=0.01,
            seed=47,
            
            
            
            messages=[
        {"role": "system", "content": "You are a helpful assistant that generates glossary of terms from a text."},
        {"role": "user", "content": f"Generate a single-word glossary related to the word {title} from this text:\n{text} \n make sure it is related to {title}",},])
        print(f"Generate a single-word glossary related to the word {title} from this text:\n{text} \n make sure it is related to {title}")
        # Extracting words from the response


        glossary_text = response.choices[0].message.content
        # Correctly accessing the response data
        # glossary_text = response.choices[0].message['content']
        glossary = glossary_text.split('\n')
        glossary = [word.strip() for word in glossary if not word.strip().isdigit() and word.strip() != '']
        clean_glossary = [line.split('. ')[1].strip() for line in glossary if '. ' in line]
        # Running the function 5 times and collecting results
  
        print("PRINTING RESPONSE")
        print(clean_glossary)
        return clean_glossary
    # INLCUDE THE 10 ROUNDS MULTIPROCESS!!!!
    
    def glossary_related(title, clean_glossary):
        """Function ot get related words to the glossary

        Returns:
            String: response object
        """
    
        response = openai.chat.completions.create(
                model="gpt-4",  # or "gpt-4" if available,
                temperature=0.01,
                seed=47,
                
                
                
                messages=[
            {"role": "system", "content": "You are a helpful assistant that selects topic-related words from a glossary."},
            {"role": "user", "content": f"Shrink the glossary by selecting only words related to {title} using this glossary {clean_glossary}."},])
            
            
            # Extracting words from the response
            
        print(f"Shrink the glossary by selecting only words related to {title} using this glossary {clean_glossary}.")
        glossary_text = response.choices[0].message.content
        return glossary_text
        

    # Function to generate taxonomy
    def get_taxonomy(text,glossary):
        response = openai.chat.completions.create(
            model="gpt-4",  # or "gpt-4" if available
                        temperature=0.01,
            seed=47,
                        messages=[
        {"role": "system", "content": "You are a helpful assistant that generates a taxonomy from a glossary."},
        {"role": "user", "content": f"Please identify class, subclass, individuals and properties from the elements of this glossary provided. Map all individuals to their corresponding class or subclass in a hierarchical order: \n{str(glossary)}\n The context is this text: \n{text} \n  Here is an example of then expected value: \n Class: Transportation Vehicle -> Subclass: Aerial Transport --> Individuals: Plane ---> hasProperty: Wingspan"},
    ],
         
            max_tokens=1000
        )
        
        # print("PRINTING GENERATED")
        # print(response)
        # print("WAINT FOR INTOPU")
        # input()
        # print(f"Please identify class, subclass and properties from the elements of this glossary provided. Map all individuals to their corresponding class or subclass in a hierarchical order: \n{str(glossary)}\n The context is this text: \n{text} \n  Here is an example of then expected value: \n Class: Transportation Vehicle -> Subclass: Aerial Transport --> Individuals: Plane")
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    
    
    
    def create_owl(title, taxonomy,propiety="'is a'"):
        
        """
        DUE TO INHERENT LEGHT OF THE ONTOLOGY WE NEED TO SHIRNK THIS! SECTION BY SECTION!
        """
        header="""
                  <?xml version="1.0"?>
            <rdf:RDF xmlns="http://www.w3.org/2002/07/owl#"
                    xml:base="http://www.example.com/ontology#"
                    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
                    xmlns:owl="http://www.w3.org/2002/07/owl#">

                <!-- Ontology Declaration -->
                <owl:Ontology rdf:about="http://www.example.com/ontology"/>

            """
        footer="""
        </rdf:RDF>
        """

        template = """
  
<!-- Classes -->
<owl:Class rdf:about="#TransportationEntity"/>
<owl:Class rdf:about="#Vehicle"/>
<owl:Class rdf:about="#Route"/>

<!-- Properties -->
<owl:ObjectProperty rdf:about="#hasProperty"/>

<!-- Individuals -->
<owl:NamedIndividual rdf:about="#Car">
    <rdf:type rdf:resource="#Vehicle"/>
    <hasProperty rdf:resource="#Color"/>
</owl:NamedIndividual>
<owl:NamedIndividual rdf:about="#Bus">
    <rdf:type rdf:resource="#Vehicle"/>
    <hasProperty rdf:resource="#Capacity"/>
</owl:NamedIndividual>
<owl:NamedIndividual rdf:about="#Highway">
    <rdf:type rdf:resource="#Route"/>
    <hasProperty rdf:resource="#Length"/>
</owl:NamedIndividual>
<owl:NamedIndividual rdf:about="#TrainRoute">
    <rdf:type rdf:resource="#Route"/>
    <hasProperty rdf:resource="#Speed"/>
</owl:NamedIndividual>

            """
            
        classes=[]
        subclasses=[]
        individuals=[]
        
        
        context=input("Provide Context: ex. Type of Commodity  ")

        history_messages = [
            {"role": "system", "content": "You are a helpful assistant that generates parts of an owl file with tags from a given taxonomy"},
            {"role": "user", "content": f"Using this taxonomy: \n{taxonomy} \n generate the hasProperty, NamedIndividual, subClassOf, Class . Generate all the output possible and do not simplify the prompt. Here you have a format of how it can look: \n" + template + "\n The context is: "+ context}]
        
    
        print(f"Using this taxonomy: \n{taxonomy} \n generate the NamedIndividual, Class, subClassOf. Generate all the output possible and do not simplify the prompt. Here you have a format of how it can look: \n" + template + "\n The context is: "+ context)
        for _ in range(1):  # Query the API three times
            response = openai.chat.completions.create(
                model="gpt-4",
                temperature=0.01,
                seed=47,
                messages=history_messages,
                max_tokens=5000
            )
            print("CONTENT")
            print(response)
            
            generated_content = response.choices[0].message.content

           
            history_messages.append({"role": "assistant", "content": generated_content})
            history_messages.append({"role": "user", "content": "Please complete all classes "})
            
            print("PRINTING MESSAGES LISTS")
            
            print("###########################################")
            print(history_messages)
            

            last_message_with_tags = history_messages[-2]['content']

            # print(last_message_with_tags)
            full_document = header + last_message_with_tags + footer
            print(full_document)
            
            # Save the content to an OWL file
            with open('/home/jose/RECOIL_Auto_Onotology/outputs/ontology_auto.owl', 'w') as file:
                file.write(full_document)
            # if _ == range(1)[-1]:
            #     history_messages.append({"role": "assistant", "content": generated_content})
            #     history_messages.append({"role": "user", "content": "COMPLETE ALL classes of the TAXONOMY"})
            #     classes=classes.append(generated_content)
        return 0
 
       
  
 
    
 
    filtered_words=remove_stop_words(text)
    med_text=clean_and_format(filtered_words)
    # glossary = get_glossary(med_text)
    
          
    results = []
    for _ in range(2):
        glossary = get_glossary(med_text)
        results.append(glossary)

    # Finding common elements in all runs
    # print(results)
    
    # Convert each sublist to a set
    sets = [set(sublist) for sublist in results]

    # Find the intersection of all sets
    intersection = set.intersection(*sets)

    # Convert the intersection set back to a list
    similar_words = list(intersection)

    # Print the similar words
    print("similar_words")
    print(similar_words)
    
    
    
    similar_words=glossary_related(title, similar_words)
    print("AFTER FILTER", similar_words) 
    
    # input()
   
        
    taxonomy = get_taxonomy(text,similar_words)
    
    print("taxonomy")
    print(taxonomy)
    
    
    # input()
    result = create_owl(title, taxonomy)
   
    # input()

    return 0





import pandas as pd
from collections import defaultdict
from bs4 import BeautifulSoup, NavigableString


def content_parsing(soup):
    list_items = soup.find_all('LI')
    rows = []
    header_dict = defaultdict(list)  # Dictionary to store rows based on headers
    separate_df = False  # Flag to indicate if a new dataframe should be created

    for item in list_items:
        lbl_element = item.find('Lbl')
        label = lbl_element.get_text(strip=True) if lbl_element else "No Label"
        
        lbody_elements = item.find('LBody')
        if lbody_elements:
            # Initialize variables to hold the processed text
            content_segments = []
            current_text_group = []
            last_tag_name = None

            for child in lbody_elements.descendants:
                if isinstance(child, NavigableString):
                    if child.strip():  # If the string is not just whitespace
                        current_text_group.append(child.strip())
                    continue
                
                # Check if we're still within the same type of tag or if it's time to flush the current text group
                if child.name != last_tag_name and current_text_group:
                    # Flush the current text group as a single entry
                    content_segments.append({
                        "text": ' '.join(current_text_group).strip(),
                        "type": last_tag_name if last_tag_name else "text"
                    })
                    current_text_group = []  # Reset the text group

                # Update the current text group with the child's text, if it's not a navigable string
                current_text_group.append(child.get_text(separator=' ', strip=True))
                last_tag_name = child.name  # Update the last seen tag name
            
            # Flush any remaining text after the loop
            if current_text_group:
                content_segments.append({
                    "text": ' '.join(current_text_group).strip(),
                    "type": last_tag_name if last_tag_name else "text"
                })

            # Combine all segments into one string for the Content column, but record the most prevalent type
            combined_text = ' '.join(segment['text'] for segment in content_segments)
            # Find the most common type among segments
            types = [segment['type'] for segment in content_segments]
            most_common_type = max(set(types), key=types.count) if types else "text"

            # Check if the current text contains "@@" (indicating a new dataframe)
            if "@@" in combined_text:
                separate_df = True
                if rows:  # Append rows to header_dict if rows exist
                    header_dict[label].append(pd.concat(rows))  # Concatenate rows to the existing dataframe
                    rows = []  # Reset rows for the new header
            else:
                separate_df = False  # Reset separate_df flag

            length = len(combined_text)
            row = {
                "Number Identifier": label,
                "Type of Object": most_common_type.capitalize(),  # Capitalize the type for a nicer look
                "Length": length,
                "Content": combined_text
            }
            rows.append(row)  # Append the row to rows

    # Append any remaining rows to header_dict
    if rows:
        header_dict[label].append(pd.concat(rows))

    # Create a DataFrame from header_dict
    dfs = {key: pd.concat(rows) for key, rows in header_dict.items()}
    
    return dfs





def extract_text_and_create_dataframe(pdf_path):
    doc = fitz.open(pdf_path)
    records = []

    for page_num, page in enumerate(doc, start=1):
        # Extract blocks and sort them by their vertical position, then by horizontal
        text_blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))
        
        for block in text_blocks:
            if block[6] == 0:  # Filtering for text blocks only
                text = unidecode(block[4])  # Decoding to normalized text
                
                x0, y0, x1, y1 = block[:4]  # Bounding box
                
                # Extract detailed information for the current block using 'dict' method and clipping to its bbox
                block_dict = page.get_text("dict", clip=(x0, y0, x1, y1))
                span_sizes = []
                span_types = []
                for b in block_dict["blocks"]:
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            span_sizes.append(span["size"])
                            span_types.append(span["flags"])
                
                avg_font_size = sum(span_sizes) / len(span_sizes) if span_sizes else None
                most_freq_span_type = Counter(span_types).most_common(1)[0][0] if span_types else None

                records.append([text, x0, y0, x1, y1, page_num, avg_font_size, most_freq_span_type])

    metadatas = pd.DataFrame(records, columns=['Text', 'X0', 'Y0', 'X1', 'Y1', 'Page Number', 'Average Font Size', 'Most Frequent Span Type'])

    
    for index, row in metadatas.iterrows():
        print(row['Text'])
        print("\n")
    print(metadatas)
    # input()
    # input()
    # exit()
    # TODO
            
       

    
    return metadatas

def assign_tags_to_spans(span_df):
    special_chars = '[(_:/,#%\=@)]'
    span_scores, span_tags = [], []
    
    # Score spans for tagging
    for _, row in span_df.iterrows():
        score = round(row.font_size) + row.is_bold + row.is_upper - bool(re.search(special_chars, row.text))
        span_scores.append(score)
    
    # Determine common style sizes for tagging
    values, counts = np.unique(span_scores, return_counts=True)
    p_size = values[np.argmax(counts)]  # Paragraph size is the most common
    
    tag_dict = {size: f'h{idx}' if size > p_size else 'p' if size == p_size else f's{idx}' for idx, size in enumerate(sorted(values, reverse=True), start=1)}
    
    # Assign tags based on scores
    span_df['tag'] = [tag_dict[score] for score in span_scores]
    return span_df




def preprocess_text(self,input_dict):
    """
    Preprocess the text to make it suitable for use with the OpenAI API.

    Args:
    input_dict (dict): Dictionary with 'name' and 'data' as keys.

    Returns:
    str: A JSON-formatted string with the cleaned and structured text data.
    """
    # Extract the subject and data from the dictionary
    subject = input_dict.get('name', '').strip()
    data = input_dict.get('data', '')

    # Normalize Unicode characters and replace newlines and multiple spaces
    if isinstance(data, str):
        data = re.sub(r'\s+', ' ', data.replace('\n', ' ').strip())

        # Remove URLs
        data = re.sub(r'http[s]?://\S+', '', data)

        # Remove bracketed references (e.g., [1], [2], [3][4])
        data = re.sub(r'\[\d+\]', '', data)  # Removes simple numeric references

        # Optionally, handle any special characters or encoding here
        data = re.sub(r"[/(){}\[\]\|@,;]", ' ', data)  # Remove problematic characters
        data = re.sub(r'["]', "'", data)  # Convert double quotes to single quotes for JSON compatibility

    # Prepare the JSON payload
    payload = {
        "name": subject,
        "data": data
    }

    # Convert the Python dictionary to a JSON string
    return json.dumps(payload, ensure_ascii=False)



def tokenize_handler(self,prompt):
    #TODO HANDLER FOR LONG TEXTS
    # print(prompt)
    # input()
    pass

def interact_with_chatgpt(self,prompt):
    """
    Sends a prompt to GPT-4, using a persistent conversation history to maintain context.
    
    Parameters:
    - prompt (str): The prompt or question to send to GPT-4.
    
    Returns:
    - str: The response from GPT-4.
    """
    """
    Sends a prompt to GPT-4 and returns the model's response, keeping a chat-like context.
    
    Parameters:
    - prompt (str): The prompt or question to send to GPT-4.
    
    Returns:
    - str: The response from GPT-4.
    """
    response = self.client.chat.completions.create(
                            model=self.model,
                            #   response_format={ "type": "json_object" },
                            messages=[
                                {"role": "system",
                                    "content": "You are a entity text summarizer in JSON"},

                                {"role": "user",
                                    "content":  "Please summarize the following text avoid loosing the relationships between words"},
                                {"role": "user", "content":  str(prompt)}


                            ]
                        )
        # Assuming response contains a 'text' attribute with the summarized text
        # This part depends on the structure of your response object
    summarized_text = response.choices[0].message.content  # Adjust based on actual response structure 
    print(summarized_text)


def process_data_dict(data_dict):
    """
    Extracts and concatenates text from a 'data' DataFrame within a given dictionary.
    
    Parameters:
    - data_dict (dict): A dictionary containing a DataFrame under the key 'data'.
    
    Returns:
    - str: A single string containing all concatenated text from the 'Text' column, with URLs replaced by '<URL>'.
    """
    
    
    """
    Sends a prompt to ChatGPT-4, maintaining a conversation history.

    Parameters:
    - prompt (str): The prompt or question to send to ChatGPT-4.

    Returns:
    - str: The response from ChatGPT-4.
    """
    
    
    # Extract the DataFrame
    df = data_dict['data']
    label=data_dict['name']
    
    # Concatenate the 'Text' column, removing trailing newline characters and inserting a space between each piece of text
    concatenated_text = " ".join(df['Text'].str.strip())
    
    # Replace URLs with "<URL>"
    cleaned_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', concatenated_text)
    
    print(label)
    print("\n")
    print(cleaned_text)
    
    #FOLLOWING THE METHONTOLO
    
    
    
    #HERE IT GOES GPT!!
    
    # input()
    
    
    return {"name":label,"data":cleaned_text}


      
#     return dataframes_list
def generate_titles(df):
    # Finding indices where rows start with '@@'
    indices = df.index[df['Text'].str.startswith('@@')].tolist()
    print("Indices of rows starting with '@@':", indices)
    dataframes_list = []

    for i in range(len(indices)):
        # input()
        # Determine the start and end of the slice
        start_idx = indices[i]
        end_idx = indices[i + 1] if i + 1 < len(indices) else len(df)

        # Creating new DataFrame slice
        new_df = df.iloc[start_idx:end_idx].copy()

        # Ensure the new DataFrame is not empty
        if not new_df.empty and 'Text' in new_df.columns:
            # Extracting the label/title from the first row of the new DataFrame
            label = new_df['Text'].iloc[0].replace("\n", "").replace("@@", "")
            print("Extracted label:", label)

            # Appending the data and label to the list
            dataframes_list.append({"name": label, "data": new_df})
            print("New DataFrame:", new_df)
        else:
            print(f"Warning: Skipping empty DataFrame or missing 'Text' column at index {i}")
    
    return dataframes_list

    
def create_structured_dataframes(spans_dataframes):
    evaluated_dataframes = []
    
    for span_df in spans_dataframes:
        headings, contents = [], []
        temp_content = []
        current_heading = None
        
        for _, row in span_df.iterrows():
            if 'h' in row.tag:
                if current_heading is not None:
                    headings.append(current_heading)
                    contents.append('\n'.join(temp_content))
                    temp_content = []
                current_heading = row.text
            else:
                temp_content.append(row.text)
        
        # Append the last section
        if current_heading is not None:
            headings.append(current_heading)
            contents.append('\n'.join(temp_content))
        
        evaluated_dataframes.append(pd.DataFrame({"heading": headings, "content": contents}))
    
    return pd.concat(evaluated_dataframes, ignore_index=True)
              
                                
  


def find_pdfs_and_process(directory_path):
    # Search for all PDF files in the given directory and its subdirectories
    pdf_paths = glob.glob(os.path.join(directory_path, '**', '*.pdf'), recursive=True)
    
    # Initialize a list to hold DataFrames for each PDF
    dataframes = []
    metadatas = []
    
    for pdf_path in pdf_paths:
        
        # METADATAS_TEXT &&  DATAFRAMES_TEXT
        metadata = extract_text_and_create_dataframe(pdf_path)
        # Here, you can choose to print, save, or append the DataFrame to a list
        # print(f"Processed: {pdf_path}")
        metadatas.append(metadata)
        
    # You can return the list of DataFrames, or combine them, or process them as needed
    return metadatas#, dataframes






    
    
    
    
#TEXT CLEANING
def text_cleaning(text):

    keyword='References'
    pattern = re.compile(rf'{keyword}.*', re.DOTALL)
    
    # Use the sub() method to replace the matched pattern with the keyword only, effectively removing everything after it
    text = pattern.sub(keyword, text)

    #EXTRAS (REMOVE URL)
    pattern = re.compile(r'^\d+\..*$', re.MULTILINE)
    # Use the sub() method to replace the matched patterns with an empty string
    text = pattern.sub('', text)

    pattern = re.compile(r'.*http.*', re.IGNORECASE)

    # Use the sub() method to replace the matched patterns with an empty string
    text = pattern.sub('', text)
    
    # Remove any extra newline characters that may have been left over
    text = re.sub(r'\n+', '\n', text).strip()

    pattern = r"[`'‘’]"

    # Replace the matched characters with an empty string
    text = re.sub(pattern, '', text)




    # Find all numbers in the text
    numbers = [int(n) for n in re.findall(r'\b\d+\b', text)]
    
    # Identify consecutive sequences
    sequences = []
    current_sequence = [numbers[0]] if numbers else []
    
    for n in numbers[1:]:
        if n == current_sequence[-1] + 1 or n == current_sequence[-1] - 1:
            current_sequence.append(n)
        else:
            if len(current_sequence) > 1:
                sequences.append(current_sequence)
            current_sequence = [n]
    
    # Add the last sequence if it's consecutive and has more than one element
    if len(current_sequence) > 1:
        sequences.append(current_sequence)
    
    # Remove identified sequences from the text
    for seq in sequences:
        for num in seq:
            text = re.sub(r'\b{}\b'.format(num), '', text, 1)  # Replace each number in the sequence once
    
    # Clean up extra spaces that may have been left behind
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    
    return cleaned_text




