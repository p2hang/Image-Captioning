import sys
import json
from nltk.tokenize import word_tokenize

def preprocess():
  filenames = ["train", "val"]
  for fn in filenames:
    path = "data/annotations/captions_" + fn + "2014.json"
    fr = open(path, 'r')
    data = json.load(fr)
    fw = open("data/processed/" + fn + "_annotations.txt", 'w')
    for item in data['annotations']:
      image_id = item['image_id']
      caption = item['caption']
      caption_words = word_tokenize(caption)
      caption = " ".join(caption_words)
      sentence = ("%d\t"% image_id + caption + "\n" )
      fw.write(sentence)
    fw.close()
    fr.close() 
  print("Preprocessing completed.")



if __name__ == "__main__":
  preprocess()