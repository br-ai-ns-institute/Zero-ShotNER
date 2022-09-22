#from scispacy.abbreviation import AbbreviationDetector
import spacy

class AbbreviationProcessor:
    """
    The class for dealing with abbreviation. It is class based on Spacy and have 2 functions, one returning abbreviations
    and the other replacing abbreviations in the text where needed
    """
    def __init__(self):
        """
        Initialization function that loads models
        """
        self.nlp = spacy.load("en_core_sci_lg")
        self.nlp.add_pipe("abbreviation_detector")

    def find_abbreviations(self,text:str) -> object:
        """
        Function that finds all abbreviations in a given text
        :param text:
        :return: A list of objects comining from spacy with all properties about every found abbreviation
        """
        doc = self.nlp(text)
        return doc._.abbreviations

    def expand_abbreviations(self,text : str) -> str:
        """
        Given a text, it replaces all abbreviations that can be found and resolved with the expanded form
        :param text: Text with abbreviations in it
        :return: Returns a text with expanded abbreviation
        """
        doc = self.nlp(text)
        abbreviations = doc._.abbreviations
        substrings = []
        remaining = text
        abbreviations = sorted(abbreviations)
        offset = 0
        for abrv in abbreviations:
            before_abr = remaining[0:abrv.start_char-offset]
            start = abrv.start_char-offset-len(str(abrv._.long_form))-10
            if start < 0:
                start = 0
            if str(abrv._.long_form).lower() in remaining[start:abrv.start_char-offset].lower():
                abbr_form = ''
                if before_abr[-1] =='(':
                    before_abr = before_abr[:-1]
            else:
                abbr_form = str(abrv._.long_form)
            substrings.append(before_abr)
            substrings.append(abbr_form)
            remaining = remaining[abrv.end_char-offset:]
            if len(remaining)>1 and remaining[0] == ')':
                remaining = remaining[1:]
                offset = abrv.end_char + 1
            else:
                offset = abrv.end_char
        substrings.append(remaining)
        new_text = ''
        for substring in substrings:
            new_text = new_text + substring
        return new_text.replace("  "," ")
