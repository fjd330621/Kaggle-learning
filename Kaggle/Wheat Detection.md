Kaggle介绍：
=====================

https://www.kaggle.com/
Kaggle成立于2010年，是一个进行数据发掘和预测竞赛的在线平台。目前网上关于Kaggle的介绍和教学已经非常多了，知乎、CSDN、简书等知识共享平台上都有，这里就不多做介绍。   
Kaggle一周能够免费提供30小时的算力时间，经过初略的估算（xjb估），其计算速度应当大于RTX2080,略等同于RTX2080TI。  
对于硬件略显不足但是又想参加比赛（特别是CV方向深层网络的比赛）的同学来说是在是一项难得福音。  

Global Wheat Detection  
=====================

_Can you help identify wheat heads using image analysis?_  
20200806
比赛已完，排名一直在1400--1500之间徘徊，不得不感叹，太难了，要学的太多了。一些直接的思考已经写在了https://www.kaggle.com/fengjiedong/begginner-thought 当中。  
后面希望能够完善这一串代码吧！！！！！！！！！！！！

# 一、代码整体架构  

就目前我浅薄的经历来看，一个完整的深度学习code应当具有：数据模块、模型模块、结果模块这三个部分。
1.数据模块的主要功能是数据导入、数据切分、数据增强等，特别是其中的数据增强，对于结果有着较大的影响。
2.模型模块主要是模型导入、模型训练、优化器设置、丢失函数设置等，通常使用的都是已经组装好的库
3.结果模块较为简单，将Test数据导入到训练好的模型中，得到最终的结果并按照比赛的要求生成特定的文件，一般都是csv格式。
# 二、具体代码

### *Dataload*  


数据载入功能，在深度学习中，数据载入往往分为两个部分：训练集验证集、测试集。其中，前一部分需要载入两部分内容，包括数据本身和数据标签；而后一部分则只需要载入数据本身即可。为了便于训练测试等操作的清晰明了，Dataset类往往也根据上述要求分成两个部分。  

验证集测试集，该类需要完成输入、变换、输出三个部分的功能

* 根据CSV信息、图像地址这两部分完成图像的导入  
* 载入变换函数 ，预先编写
* 输出指定格式数据    

具体代码如下：  

class WheatDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()
		#读取csv中关于images的信息，便于后续读取
        self.image_ids = dataframe['image_id'].unique()#同一id下有多个标签，获取唯一id
        self.df = dataframe#导入CSV
        self.image_dir = image_dir#图像地址
        self.transforms = transforms#导入图片变换方式

    def __getitem__(self, index: int):#魔术方法、迭代器，index为__len__方法获取数据数量，从0开始到最大值

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]#获取该id下所有标签
		#利用OpenCV库导入图像
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)#格式转换
        image /= 255.0#均一化处理
		#读取具体坐标并将其中pandas格式转化为numpy格式
        boxes = records[['x', 'y', 'w', 'h']].values
        #将coco格式的目标坐标格式转为为pascal格式
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        #计算目标面积。。额，忽然感觉这一步有点愚蠢
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)#新内存下建立tensor格式

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}#初始化目标字典
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd
		#对图像和目标同时进行变化，如翻转、裁剪等
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            #zip(*sample['bboxes'])转置
            #map()转换为torch.tensor格式
            #tuple构建元组
            #以xxxxx
             # yyyyy
              #aaaaa
             # bbbbb
             # 为格式
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]
        
 ### *Pseudo Label*

初次接触伪标签这一概念，目前并不能从原理上很好的理解之一方法，但是在今天FasterRCNN的实践当中，PL确实能够在一定程度上提升LB的成绩。

对于深度学习而言，拥有标准Label的数据是极其宝贵的，但是另一方面，由于DP的本质使得其对于图像数量的需求是极大的。因此，目前已经出现很多“奇奇怪怪”的方法来增加的数据集的数量。常见的方法有以翻转为基础的Image Augmentation、目前火热的迁移学习等。而Pseudo Label也是其中的一种方式。说实话，这种方式有点类似于遥感中常用的半监督分类。其具体步骤如下：

* 1.利用已有标准数据集训练模型

* 2.将无标签数据导入上面训练好的模型中获得无验证的标签数据

* 3.混合两种数据集再一次训练模型  

  经过上述步骤能够一定程度上提升模型精度  

  不过，目前我并不能很好的从原理上理解使用这一方法及其效果的原因。

### *Augmentation* 

调用了Kaggle一个notebook中的图像增强方式，目前只是单纯的调用，并没有从源代码处理解其方法和原理，希望后续有时间来细读这一串code。

遇到的一个问题是提供的数据输出方式并不能直接导入到model当中，需要经过转化，目前想到的方式有三：

* 将得到的图片和标签先导出到本地，再通过已有的读取方式进行读取。这种方式思路简单但是实现繁琐，耗费内存。
* 修改目前的Loader，将转换后的图片模仿已有的loader构建新的sample，目前正在尝试当中
* 构建全新的loader，将上一步得到的新的图像作为新的数据集导入   

遇到问题的原因：在利用Pytorch进行深度学习的操作时，常常遇到的问题便是格式的转化，模型需要tensor格式数据，而我们在操作的时候往往使用的是numpy下array格式的数据，对于这一部分的理解还不是非常的深刻，转化是需要注意的点也没有非常的清楚，需要后续构建清晰的理解。
最后还是通过更改Loader进行了数据增强，主要使用了Mosaic，当然这不是我们常说的马赛克，而是对已有数据集图像进行分割和拼接，以此来获得更多的数据集进行训练。
另外，https://albumentations.ai/docs/ 库也可以对图像进行从初步的图像处理
