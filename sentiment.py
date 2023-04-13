import re

def makesentimentdic():    
    sentiment_dict = {}
    with open('/data/minki/swbd/subjclueslen1-HLTEMNLP05.tff', 'r', encoding='utf-8') as f:
        # sentiment_dict = { re.split(" =\n", line) for line in f.readlines()}
        for line in f.readlines():
            line = re.split("=| |\n", line)
            if line[11] == 'neutral':
                sentiment_dict[line[5]] = 0
            elif line[11] == 'positive':
                if line[0] == 'strongsubj':
                    sentiment_dict[line[5]] = 2
                else:
                    sentiment_dict[line[5]] = 1
            elif line[11] == 'negative':
                if line[0] == 'strongsubj':
                    sentiment_dict[line[5]] = -2
                else:
                    sentiment_dict[line[5]] = -1
    return sentiment_dict

if __name__ =="__main__":
    a=makesentimentdic()
    print(a)