# from utils import * #IS A 
from utils import * # PROPERTY OF

"""
   *-------------------------------------------------------------------------------------
   | Name:      Jose Tupayachi                                                        
   | HW:    gptOntoWorkflow                                                               
   |------------------------------------------------------------------------------------|
   | Purpose:   Testbed for GTP Workflow Model
   |
   *-------------------------------------------------------------------------------------
"""



class LlmOntoWorkflow():
    def __init__(self,input,model,output_folder,api_key,title):
        self.data = None
        self.sentences = None
        self.classes_propietes = None
        self.herarchy_structure = None
        self.model = model
        self.turtle_tiples = None
        self.input=input
        self.output_folder=output_folder
        # self.messages=None
        self.api_key=api_key
        self.title=title

    def preparation(self, format, url=None):
        """
        Input: Start with a .txt file containing the text you want to transform into a knowledge graph.
        Text Cleaning: Clean the text to remove any irrelevant information, such as stopwords, punctuation, and formatting issues.
        Sentence Segmentation: Break the text into sentences to process them individually for information extraction.

        """





        if format == 'xml':
            print(self.input)
            # soup=process_xml_files(self.input)
            # df_extracted_texts=content_parsing(soup)
            # input()
        
        elif format == 'pdf':
            print(self.input)
            metadatas=find_pdfs_and_process(self.input)
            
            
            # dataframes=pd.concat(dataframes)
            metadatas=pd.concat(metadatas)
    
            structured_dataframes=generate_titles(metadatas)
            
        self.data=structured_dataframes

        

    def llm_processing(self):
        """
        Named Entity Recognition (NER): Identify and classify named entities in the text into predefined categories like persons, organizations, locations, etc.
        Relationship Extraction: Identify relationships between the named entities discovered in the previous step. This involves understanding how entities are connected through verbs or prepositional phrases.
        Entity Disambiguation: Resolve ambiguities in entity identification, ensuring that each entity is uniquely identified and consistent across the text.

        """

        process_data_dict_list=[]
        print(self.data)
        for i in self.data:
            print(i)
            print("cgcgcgc")
            # input()
            item=process_data_dict(i)
            process_data_dict_list.append(item)
            

        openai.api_key = self.api_key
        


        #WE HAVE CLEANED TEXTS WITH AN STRUCTURE:
        for dict in process_data_dict_list:
            
            
            text=preprocess_text(self,dict)
            # print(text)
            # chunks = chunk_text_by_sentences(text, sentences_per_chunk=5)
            title=self.title 
            pre_ontology = generate_ontology(text, title)
            print(pre_ontology)
            print("#@@@@@@@@@@@@@@@@@@@@@@")
            # create_owl_file(pre_ontology['Title'], pre_ontology['Glossary'],pre_ontology['Taxonomy'],pre_ontology['Relationship Mapping'])
            # print("################################")
            
            
  
        pass
            
       
    def owl_serialization(self):
        """
        Model the extracted information as RDF (Resource Description Framework) triples, which consist of subject, predicate, and object. 
        Each triple represents a fact or a relationship from the text.

        Subject: Represents the entity.
        Predicate: Represents the relationship between entities.
        Object: Represents the entity that is linked to the subject by the predicate.

        Turtle Serialization: Serialize the RDF triples into the Turtle (Terse RDF Triple Language) format. Turtle is a textual syntax for RDF that expresses data in triples, making it easier to read and write by humans.
        """
        pass


if __name__ == "__main__":

    ROOT_PATH='/home/jose/RECOIL_Auto_Onotology'
    FORMAT = 'pdf'
    CLASSES_PROP = 'outputs/classes_propietes.pkl'
    TYPE = ''  # ''
    
    API_KEY = ''
    # openai.api_key = API_KEY


    INPUT_FOLDER = input("Enter Input Folder: ex. '/inputs/FAF_SHORT/'   ")
    TITLE=input("Enter title: ex. 'Freight Analysis Framework'  ")
    

    obj = LlmOntoWorkflow(input=ROOT_PATH+INPUT_FOLDER,model="gpt-4"
                          ,output_folder=ROOT_PATH+'/.tmp/', api_key=API_KEY,title=TITLE)
    # /Users/user/RECOIL_Auto_Onotology/inputs/FAF5-User-Guide.pdf

    if TYPE == 'SUMMARIZE':

        print("AFTER SUMMARIZE")
        with open(ROOT_PATH+'/.tmp/summarized_data.pickle', 'rb') as f:
            process_data_dict_list = pickle.load(f)
        print(process_data_dict_list)


    else:
        obj.preparation(FORMAT)
        obj.llm_processing()
        
