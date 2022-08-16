import copy
from copy import deepcopy
origin = [1,2,[3,4]]
# origin里面有三个元素，1,2,[3,4]
copy1 = copy.copy(origin)
copy2 = deepcopy(origin)
print(copy1 == copy2)   # True
print(copy1 is copy2)   # False
# copy1 和copy2看上去相同，但已经不是同一个object
origin[1] = "hey!"
print(origin)   # [1, 2, ['hey!', 4]]
print(copy1)    # [1, 2, ['hey!', 4]]
print(copy2)    # [1, 2, [3, 4]]
