
from docx.shared import Inches
from docx import Document
def dell(output):    
    document = Document()
    
    document.add_heading('Hand Drawn', 0)
    
    #document.add_picture('flowchart/output/output.jpg', width=Inches(1.25)) 
    with open('output/output3.txt','r') as f:
        for line in f:
               print(line)       
               document.add_paragraph(line)
               #document.add_paragraph('first item in ordered list', style='ListNumber')
    document.save(output)
