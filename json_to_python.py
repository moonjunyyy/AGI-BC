import json

utterances             = []
backchannel_responses  = []
backchannel_categories = []


with open('backchannel_results.json', 'r') as f: bc_jsonfile = json.load(f)
results ={
    'samples': [
    ],
}
for conversations in bc_jsonfile['conversation']['turns']:
    utterance            = conversations['utterance']['korean']
    backchannel_response = conversations['backchannel_response']['korean']
    backchannel_category = conversations['backchannel_response']['category']
    print(f"Utterance: {utterance}\nBackchannel Response: {backchannel_response}\nBackchannel Category: {backchannel_category}\n")
    results['samples'].append({
        'utterance': utterance,
        'backchannel_response': backchannel_response,
        'backchannel_category': backchannel_category,
    })
with open('backchannel_generated_text.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)