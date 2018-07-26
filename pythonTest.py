print("Hello Haifeng")

print(True and False)

fruits = ['apple','peach','mango']
for fruit in fruits:
    print ('当前水果:', fruit)
    
i = 1
while i<10:
    i+=1
    if i%2 == 0:
        print (i)
        continue

"列表推倒"
multiples = [i for i in range(30) if i%3 is 0 ]
print(multiples)

multiples = []

for i in range(30):
    if i%3 is 0:
        multiples.append(i)
print(multiples)

#求五个收盘价的最大值和最小值
close=[20.5,33.6,58.3,12.6,10]
max_num = max(close)
print('最高收盘价为：',max_num)
min_num = min(close)
print('最低收盘价为：',min_num)

#从字典中选出PE低于某个标准的个股
def pick_by_pe(stock_data,pe):
    
    stock_list = []
    for key in stock_data :
        if stock_data[key] < pe :
           stock_list.append(key)
    return stock_list

stock_data = {'00238':12.3,'60012':14.5,'9002':56.3}
stock_pe = pick_by_pe(stock_data,50)
print(stock_pe)
            

is_one = lambda x: print('yes') if x==1 else print('no')

is_one(1)

#Python的类
class House(object) :
    houseType = "studio"
    houseSize = "42m"
    
    def watchTV(self):
        print("I'm watching TV")
    def sleep(self):
        print("I go to sleep")
    @staticmethod
    def staticSleep():
        print("I'm static sleep")
    @classmethod        
    def classSleep(self):
        print("I'm class method")
        
house = House()
house.watchTV()
print(house.houseSize)

House.staticSleep()
house.classSleep()
House.classSleep()

#Python的类的继承，多态
class ChildHouse(House):
    houseType = "grand maison"
    houseSize = "88m"
    def watchTV(self):
        print("I'm watching TV in childHouse")
        
houseChild = ChildHouse()
houseChild.watchTV()