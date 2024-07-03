import openai
from openai import OpenAI
import os
import json
import pandas as pd
import time
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

# Add the arguments
parser.add_argument('--start', type=int, help='start index')
parser.add_argument('--end', type=int, help='end index')

# Parse the arguments
args = parser.parse_args()



os.environ['OPENAI_API_KEY'] = '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'
client = OpenAI()
df = pd.read_excel('datasets/faq_train_4k.xlsx')
# df = pd.read_json('data/faq_train.json', orient='records', lines=True).sample(2000)
# exs = """
# Passage 1: cacbon giúp gia tăng hàm lượng serotonin điều hòa tâm trạng học một sở thích những người có sở thích riêng sẽ thư giãn và ít bực bội hơn sở thích giúp mọi người tránh bị trong cuộc sống ăn ngoài những ai ra ngoài ăn với bạn bè sẽ ít bị căng thẳng và trầm cảm hơn một nghiên cứu tại đức tìm thấy ăn ngoài có tác dụng như một cách thoát
# Summary 1: Cacbon giúp tăng hàm lượng serotonin giúp điều hoà tâm trạng

# Passage 2: cách chữa mẹo nếu không thành công thì cũng bỏ qua họ chỉ đến bệnh viện khi mẩu xương đã gây biến chứng sự khinh suất này đã dẫn đến nhiều hậu quả nghiêm trọng như thủng hoặc viêm thực quản nhiễm trùng máu tử vong nguyên nhân gây hóc xương thường gặp là thói quen ăn cả thịt lẫn xương vừa ăn vừa nói chuyện cười đùa hoặc ăn uống trong lúc say rượu bệnh nhân đang ăn thấy nuốt nuốt vướng tăng tiết nước bọt có thể không ăn uống tiếp được nếu xương
# Summary 2: Việc ăn thịt lẫn xương và ăn uống có thể gây hóc xương, gây thủng thực quản hoặc nhiễm trùng máu
# """

exs = """
Passage 1: Chào các bác sĩ . Con mình được 1 tháng rưỡi . Bé rất hay thức đêm ngủ ngày . Ban đêm 1 lần thức kéo dài 6-7 tiếng . Có khi bé ngủ đến 8h tối dậy và thức đến 7h sáng ngày hôm sau dù mình có làm cách nào bé vẫn không chịu ngủ . Dạo gần đây bé khan tiếng và thức đêm ngủ ngày thường xuyên hơn . Nếu bé thức như vậy có ảnh hưởng đến sức khoẻ không ạ và làm cách nào để cải thiện ạ ?
Summary 1: bé hay thức đêm

Passage 2: Con em được 4 tháng tuổi em có nên tiếp tục vệ sinh mắt mũi bé hàng ngày bằng nước muối sinh lý nữa không ạ ? Mắt bé thì ngủ dậy có tí ghèn thôi . Em nghe nói không nên lạm dụng nhiều nước muối sinh lý quá sẽ làm khô mắt bé . Vậy cho em hỏi em có nên tiếp tục nhỏ mắt , mũi cho bé hàng ngày sau khi tắm nữa không hay khi nào mắt có ghèn hay sổ mũi thì mới nhỏ ?
Summary 2: vệ sinh mắt mũi thường xuyên bằng nước muối sinh lý có tốt không ?
"""

guideline = f"""
As a professional summarizer, create a concise and comprehensive vietnamese summary of the provided passage, while adhering to these guidelines:
1. Keep the summary as short as possible without losing key information. The length of the summary is at most 20% of that of the passage (except too short passage)
2. Retain as many entities as possible as long as the limit is not exceeded
3. Retain the purpose of the passage
4. Summaries must sound natural.
5. Do not use a third-person perspective
6. Follow the format of the examples' summaries.

Examples:
{exs}
"""
#f"""Generate vietnamese summaries of these passages. Do not use a third-person perspective. The length of the summary is at most 20% of that of the passage, except for passages that are too short. Retain as many entities as possible while be concise as much as possible. Two examples are provided. Follow the format of the summaries in the examples.
def gpt_summary(passage):
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {
          "role": "system",
          "content": f"""Generate vietnamese summaries of these passages. Do not use a third-person perspective. Two examples are provided. Follow the format of the summaries in the examples.
          Examples:
          {exs}"""        
        },
        {
          "role": "user",
          "content": f"Passage: {passage}\nSummary:"
        }
      ],
      temperature=0.7,
      max_tokens=200,
      top_p=0.9
    )
    return response.choices[0].message.content

t = time.time()
l = {}
print('index'+'\t'+'transcript'+'\t'+'summary_gpt')
file_path = './faq_train_4k_gpt.tsv'

# Checking if the file exists before trying to open it
try:
    with open(file_path, 'r') as file:
        lines = file.readlines()
    result = lines
except FileNotFoundError:
    result = "The file 'text.txt' does not exist."


# Wrap the loop in a try-except block
# for i, r in df[args.start:args.end].iterrows():
# for i, r in enumerate(result):
for i, r in (df.iterrows()):
    try:
        index, passage = r['Unnamed: 0'], r['transcript']
#         index, passage = i, r
        # Use gpt_summary to summarize the passage and store it in l
        l[index] = gpt_summary(passage)
        print(str(index) + '\t' + passage + '\t' + l[index])
    except Exception as e:
        # If an error occurs, print the error message and continue with the next iteration
        print(f"Error processing index {index}: {e}")

# print('Time taken', time.time()-t)
# Save the dictionary using json, ensuring this is outside the loop to save progress regardless of errors
# with open(f'./summaries-gpt3_5-{args.start}-{args.end}.json', 'w') as file:
#     json.dump(l, file)
