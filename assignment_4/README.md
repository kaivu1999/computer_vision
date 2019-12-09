## Hand Gesture recognition   ( Kaivalya and Vignesh )

### Problem statment can be found [here]()

Classes used were
- Previous
- Next
- Stop

50 * 50 images were used to train the model.

Image examples that were used to train after background subtration, edge detection ...

![next](https://user-images.githubusercontent.com/35027192/68789112-cc2ca780-066a-11ea-8fd2-b44b8eebd727.png)
![previous](https://user-images.githubusercontent.com/35027192/68789136-d8b10000-066a-11ea-9dbb-4bc5d2e0b47a.png)


We use Sliding Window Approach to find a good bounding square around the hand. Once found a good square, we just use Tracker (Boost Method) to track the hand. The model in then trained on these 50 x 50 images (resized from 200 x 200 pixel from our bounding box )


File structre
```
├── collect_data.py
├── Data
│   ├── train
│   │   ├── next
│   │   │   ├── class1__0.png
│   │   │   └── ...
│   │   ├── previous
│   │   │   ├── class0_0.png
│   │   └── stop
│   │       ├── class2_0.png
│   └── val
│       ├── next
│       │   ├── class1__0.png
│       ├── previous
│       │   ├── class0_0.png
│       └── stop
│           ├── class2_0.png
├── inference.py
├── load.py
├── model1
├── model1_acc.png
├── model2
├── model2_accuracy.png
├── model2_loss.png
├── network.py
├── prepare_data.py
├── requirements.txt
└── slidingW_Inference.py

```

To run live demo (By default model2 gets run)
```
python inference.py
```

collect_data.py was used to make appropriate data 

load.py is the script where training occurs

model2 are the weights of the learned model with accuracy 0.98, 0.93 on train and test data


We have proposed two models

model1(4 layers) , model2 (5 layres)
### model2 is as follows
```
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3 )
        self.fc1 = nn.Linear(64*4*4, 64)  
        self.fc2 = nn.Linear(64, 3)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x) # check this
        return x
``` 


Some of the training plots are 

![model2_accuracy](https://user-images.githubusercontent.com/35027192/68789321-3b0a0080-066b-11ea-9f37-b48cb26300df.png)

![model2_loss](https://user-images.githubusercontent.com/35027192/68789323-3ba29700-066b-11ea-936f-85dc331cd10d.png)


Real Time demo :
![real_time](https://user-images.githubusercontent.com/35027192/68789505-9a681080-066b-11ea-8c25-abd6f87fef54.png)





---
The dataset of hands were made with the help of Kartik, Kailash, Vignesh, Apaar, Nipun, Madhav, Anup and myself from Karakoram hostel.

---