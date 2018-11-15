import json,glob
import collections
def load_data(json_dir):
    """ load all of the json file and save the label_class to two list  
    Args:
        json_dir: the dir  of all json file
    Returns:
        idx:       each id of json file  idx.shape = (len(json_file),),idx[0].shape = (n,)  ;[[first file],[second file],[third file]]
        classx:    each class of json file                                                  ;[[first file],[second file],[third file]]
    """
    idx = []     # 索引
    classx = []  # 类别标签
    for i in sorted(glob.glob(json_dir)):
        image_id = []
        image_class = []
        with open(i,'r') as f:
            data = json.load(f)
            for i in range(len(data)):
                image_id.append(data[i]["image_id"])
                image_class.append(data[i]["disease_class"])
            idx.append(image_id)
            classx.append(image_class)
        del image_class,image_id
    return idx,classx

def each_class(idx,classx):
    """ get corresponding every class of id for each file
    Args:
        idx:   the id of each json ;
        classx: the label of each json;
    Returns:
        results:   the result of each json file;results.shape = (len(json_file),),results[0].shape = (n,);[[,,,,],[],[],...]
        idd:       each id of json file                                                                    [,,,,]  
    """
    idd = idx[0]  # 所有的图片名字
    results = []
    for j in range(len(idx)):  # 每一文件
        result = []
        for i in idd:
            result.append(classx[j][idx[j].index(i)])  # 类别
        results.append(result)
    return results,idd
def caculate_label(results):
    """caculate the label of each json
    Args:
        results: return value of each_class
    Returns:
        goals:   final label

    """
    goal = []
    for j in range(len(results[0])):
        bit = []
        for i in range(len(results)):
            bit.append(results[i][j])  # 所有文件中,每一个id对应的标签组成一个列表
        goal.append(sorted(collections.Counter(bit).items(),key = lambda x:x[1],reverse = True)[0][0]) 
        # TODO: 如果有两个类别的次数相同,会选择类别在列表前面的那个
    return goal
def goal2json(goal,idd,final_path):
    """ convert the goal to json file
    Args:
        goal:       the final label of idd
        idd:        all data name
        final_path: generate file 
    Returns:
        None
    """
    final = []
    for i in range(len(goal)):
        dict = {}
        dict["disease_class"] = goal[i]
        dict["image_id"] = idd[i]
        final.append(dict)
        del dict
    with open(final_path,'w') as f:
        json.dump(final,f)
def caculate_diff(results,idd,index1,index2):
    """caculate  total different label on two json(index1,index2)
    Args:
        index1 :list.index = 0,1,2,3,4
    Returns:
        different total number 
    """
    diff = 0
    for j in range(len(idd)):
        if results[index1][j] != results[index2][j]:
            diff+=1
    return diff

def caculate_diff_3(path):
    file_name = "test/"
    diff = 0
    list = []
    for i in glob.glob(path):
        x = i.split("/")[-1]
        list.append(x)
    list = sorted(list)
    with open(file_name+list[0],'r') as f1:
        data1 = json.load(f1)
    with open(file_name+list[1],'r') as f2:
        data2 = json.load(f2)
    with open(file_name+list[2],'r') as f3:
        data3 = json.load(f3)
    
    for i in range(len(data1)):
        if data1[i]["disease_class"] != data2[i]["disease_class"] and data2[i]["disease_class"] != data3[i]["disease_class"] and data3[i]["disease_class"] != data1[i]["disease_class"]:
            diff += 1
            print(data1[i]["image_id"])
    return diff


if __name__ == "__main__":

    idx,classx = load_data("../Result/*.json")
    results,idd = each_class(idx,classx)
    goal = caculate_label(results)
    goal2json(goal,idd,"../Result/fusion_result.json")   # 融合不同的json文件
#    print(caculate_diff(results,idd,0,1))   # 计算两个json文件有多少个不同的label
    # print("total different {}".format(caculate_diff_3("./test/*.json"))) #　计算三个json文件都不同的个数以及id
