
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs= ["너무 재밋어요", "참 최고에요", "참 잘 만든 영화에요",
        '추천하고 싶은 영화입니다','한 번 더 보고 싶네요','글쎄요',
        '별로에요', ' 생각보다 지루해요','연기가 어색해요',
        '재미없어요','너무 재미없다','참 재밋네요']

#긍정 1 , 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

#[[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24]]
'''
앞자리를 0으로 채운다 (0은 값이 없다)

'''
from tensorflow.keras.preprocessing.sequence import pad_sequences #시퀀스를 채우겠다 (0으로채우기 위해)
pad_x = pad_sequences(x, padding='pre')             #뒤  post
print(pad_x)
print(pad_x.shape)
'''
[[ 0  0  0  2  3]  (12, 5) 
 [ 0  0  0  1  4]
 [ 0  1  5  6  7]
 [ 0  0  8  9 10]
 [11 12 13 14 15]
 [ 0  0  0  0 16]
 [ 0  0  0  0 17]
 [ 0  0  0 18 19]
 [ 0  0  0 20 21]
 [ 0  0  0  0 22]
 [ 0  0  0  2 23]
 [ 0  0  0  1 24]]
 단어 수는 24개에 0까지 해서 25
'''

word_size = len(token.word_index) + 1
print("전체 토큼 사이즈:", word_size) #25

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten,Conv2D, Conv1D

model = Sequential()
# model.add(Embedding(25,# 25 단어 사전 의 갯수 (원래는 25x 25짜리였는데 아래 벡터로 )
#                     10, #10 아웃풋 (벡터로 줄인거 임의로 줄여도 된다로드의 갯수 )
#                     input_length=5)) #5열의 갯수
# 3가지다 다르게 써도 되지만 단어 사전보다 더 높게 줘야 한다
model.add(Embedding(10, input_length=5)) #input_length 만 맞춰 주면  x의 컬럼과 일치해야 한다 

# model.add(Embedding(26, 10))  # 위에 걸 주석 처리 하고 이것으로 돌리면5가 빠져있다 (높은 수는 상관없다 (단어 사전의 갯수가 작은 값을 넣으면 터진다)
model.add(LSTM(32))
model.add(Conv1D(32, 2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])

model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1] #메트릭스 값은 1이 들어 간다

print("acc : ",acc)