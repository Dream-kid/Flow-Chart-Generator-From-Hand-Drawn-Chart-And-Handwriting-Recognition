# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:28:34 2019

@author: Imsourav
"""

from docx.shared import Inches
from docx import Document
def dell1():    
    document = Document()
    
    document.add_heading('Hand Drawn', 0)
    
    document.add_picture('flowchart/output/output.jpg', width=Inches(1.25)) 
    document.save('flowchart/output/demo.docx')