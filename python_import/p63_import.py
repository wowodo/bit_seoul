#path 가 걸려있는 아나콘다 폴더에 아래에 test 넣어 놓고 불러 올수 있나 
from test_1208 import p62_import
p62_import.sum2()

print("=================")
 

from test_1208.p62_import import sum2
sum2()