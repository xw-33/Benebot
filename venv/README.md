# Description

 main.py and vector.py uses the csv file "realistic_restaurant_reviews.csv" and acts as a test version
 main_qna.py and vector_qna.py uses the csv file "qna", which is derived from PolyU websites
 
 packages needed are showed in "requirements.txt"
 
 This project aims to enable deepseek-r1 to answer PolyU related questions by extracting information from a dataset (qna.csv)
 The structure of the dataset is built based on question-answer pairs
 
 Examine the input question
 If text then use the RAG to retrieve answers from csv
 If Image or Location then determine "entity":
  retrieve from csv + generate text response + format response.
  response = map/image + text