import matlab_extractor as me
import os

data = 'C:/Users/16098/Documents/Research Project/Exported_Data_l_10/'

def reader(data, method, length):
    content = os.listdir(data)
    ind = content.index(method + "_" + str(length) +".txt")
    print(ind)
    text_file = open(data + content[ind], "r")
    lines = text_file.read().split()
    lines = [float(i) for i in lines]
    return lines


sevcik_data = reader(data, "sevcik", 51)
print(sevcik_data)


file = 'C:/Users/16098/Documents/Research Project/MATLAB/Data_l_10_one_side/'
#xs, ys = me.extract(file,False)
#h = .0013

def indexer(xs,ys):
    my_dict = {}
    k = 0
    for i in range(0, len(xs)):
        for j in range(0, len(ys)):
            var = str(xs[i]) + "_" + str(ys[j])
            my_dict[var] = k
            k = k + 1
    return my_dict


def feature_sizer(feature,x,y,h,my_dict,bound):
    #surveys space in form of snake
    in_str = True
    var = str(x) + "_" + str(y)
    loc = my_dict[var]
    def up(x,y,h):
        return x,y+h
    def down(x,y,h):
        return x,y-h
    def left(x,y,h):
        return x-h,y
    def right(x,y,h):
        return x+h,y

    #while in_str:
     #   it = 1
      #  pass


    return

